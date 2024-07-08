
import yfinance as yf
import pandas_ta as ta
import os
from tqdm import tqdm
import argparse
import plotly.graph_objects as go
from datetime import datetime, timedelta

tqdm.pandas()

max_val = 0
min_val = 99999
structure_brake_index = []
engulfing_candle = []
end_date = datetime.now()  
start_date = end_date - timedelta(days=50)
bblength=30
bbstd=2.0

def init(calculate_structure_brake, num_bars, num_engulfing_bars):
    #df = yf.Ticker("AAPL").history(period="1d", interval="5m")
    df = yf.download("USDJPY=X", start=start_date, end=end_date, interval='5m')
    df.index = df.index.tz_convert('Europe/Rome')
    df = df.dropna()

    df["EMA"] = ta.ema(df.Close, length=200)
    df["EMA_slow"]=ta.ema(df.Close, length=50)
    df["EMA_fast"]=ta.ema(df.Close, length=30)
    df["RSI"] = ta.rsi(df.Close, length=15)
    df["ATR"] = ta.atr(df.High, df.Low, df.Close, length=10)

    my_bbands = ta.bbands(df.Close, length=bblength, std=bbstd)
    #my_bbands = ta.bbands(df.Close, length=15, std=1.5)
    df=df.join(my_bbands)
    df["EMATrend"] = df.progress_apply(lambda row: ema_signal(df, row.name, 10), axis=1)
    df["FSEMATrend"] = df.progress_apply(lambda row: fs_ema_signal(df, row.name, 10), axis=1)
    df['BBStrategy'] = df.progress_apply(lambda row: bb_strategy(df, row.name, 7), axis=1)
    
    if calculate_structure_brake:
        df.iloc[-num_bars:].progress_apply(lambda row: structure_brake(df, row.name), axis=1)
    
    if num_engulfing_bars > 0:
        df.iloc[-num_engulfing_bars:].progress_apply(lambda row: find_engulfing_candle(df, row.name), axis=1)
    
    return df

def ema_signal(df, current_candle, backcandles):
    current_index = df.index.get_loc(current_candle)
    start = max(0, current_index - backcandles)
    end = current_index + 1
    relevant_rows = df.iloc[start:end]
    
    # Drop NaN values in relevant rows
    relevant_rows = relevant_rows.dropna()
    
    if len(relevant_rows) == 0:
        return 0  # Ritorna 0 se non ci sono dati validi
    
    if all(relevant_rows["Close"] > relevant_rows["EMA"]):
        return 1
    elif all(relevant_rows["Close"] < relevant_rows["EMA"]):
        return -1
    else:
        return 0


def fs_ema_signal(df, current_candle, backcandles):
    current_index = df.index.get_loc(current_candle)
    start = max(0, current_index - backcandles)
    end = current_index + 1
    relevant_rows = df.iloc[start:end]

    if all(relevant_rows["EMA_fast"] < relevant_rows["EMA_slow"]):
        return -1
    elif all(relevant_rows["EMA_fast"] > relevant_rows["EMA_slow"]):
        return 1
    else:
        return 0

def bb_strategy(df, current_candle, backcandles):
    bbl = "BBL_"+str(bblength)+"_"+str(bbstd)
    bbu = "BBU_"+str(bblength)+"_"+str(bbstd)
    if (df.Close[current_candle]<=df[bbl][current_candle]
        and df.RSI[current_candle]<=25):
        return 1
    if (df.Close[current_candle]>=df[bbu][current_candle]
        and df.RSI[current_candle]>75):
            return -1
    return 0
  

def structure_brake(df, current_candle):
    global max_val, min_val
    row = df.loc[current_candle]
    if row["High"] > max_val:
        structure_brake_index.append([max_val, row["High"], row.name, "MAX"])
        max_val = row["High"]
    if row["Low"] < min_val:
        structure_brake_index.append([min_val, row["Low"], row.name, "LOW"])
        min_val = row["Low"]

def find_engulfing_candle(df, current_candle):
    current_index = df.index.get_loc(current_candle)
    if current_index == 0:
        return
    
    current_candle_data = df.iloc[current_index]
    previous_candle_data = df.iloc[current_index - 1]

    result = is_engulfing(current_candle_data, previous_candle_data)
    if result != 0:
        direction = "UP" if result == 1 else "DOWN"
        engulfing_candle.append([current_candle_data.name, direction])

def is_engulfing(current_candle_data, prev_candle_data):
    if (current_candle_data['Close'] > current_candle_data['Open'] and
        prev_candle_data['Close'] < prev_candle_data['Open'] and
        current_candle_data['Close'] > prev_candle_data['Open'] and
        current_candle_data['Open'] < prev_candle_data['Close']):
        return 1  # Engulfing verso l'alto

    if (current_candle_data['Close'] < current_candle_data['Open'] and
        prev_candle_data['Close'] > prev_candle_data['Open'] and
        current_candle_data['Close'] < prev_candle_data['Open'] and
        current_candle_data['Open'] > prev_candle_data['Close']):
        return -1  # Engulfing verso il basso

    return 0  # Nessun engulfing

def tradeInfoManual(df):
    print("EMA: ", df["EMATrend"].iloc[-1])
    print("FSEMA: ", df["FSEMATrend"].iloc[-1])
    print("BBStrategy: ", df["BBStrategy"].iloc[-1])
    print("RSI: ", df["RSI"].iloc[-1])
    print("ATR: ", df["ATR"].iloc[-1])
    
    if len(structure_brake_index):
        print("\nStructure Break Index:")
        for item in structure_brake_index:
            print(item)
   
    if len(engulfing_candle):
        print("\nEngulfing Candles:")
        for item in engulfing_candle:
            print(item)

def plot_chart(df):
    bbl = "BBL_"+str(bblength)+"_"+str(bbstd)
    bbu = "BBU_"+str(bblength)+"_"+str(bbstd)
    bbm = "BBM_"+str(bblength)+"_"+str(bbstd)
    df = df.iloc[-50:]
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick'
        ),
        go.Scatter(
            x=df.index,
            y=df['EMA'],
            mode='lines',
            name='EMA',
            line=dict(color='blue', width=2)
        ),
        go.Scatter(
            x=df.index,
            y=df['EMA_slow'],
            mode='lines',
            name='Slow EMA',
            line=dict(color='purple', width=2)
        ),
        go.Scatter(
            x=df.index,
            y=df['EMA_fast'],
            mode='lines',
            name='Fast EMA',
            line=dict(color='yellow', width=2)
        ),
        go.Scatter(
            x=df.index,
            y=df[bbu],
            mode='lines',
            name=bbu,
            line=dict(color='blue', width=1, dash='dash'),
            fill='tonexty',  # Riempi l'area sotto la linea
            fillcolor='rgba(0, 0, 255, 0.1)'  # Colore azzurro trasparente
        ),
        go.Scatter(
            x=df.index,
            y=df[bbl],
            mode='lines',
            name=bbl,
            line=dict(color='blue', width=1, dash='dash'),
            fill='tonexty',  # Riempi l'area sotto la linea
            fillcolor='rgba(0, 0, 255, 0.1)'  # Colore azzurro trasparente
        ),
        go.Scatter(
            x=df.index,
            y=df[bbm],
            mode='lines',
            name=bbm,
            line=dict(color='orange', width=1, dash='dash'),
        )
    ])
    
    # Aggiungi segnali di trading
    buy_signals = df[df['BBStrategy'] == 1]
    sell_signals = df[df['BBStrategy'] == -1]
    
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['Close'],
        mode='markers',
        marker=dict(color='green', size=10),
        name='Buy Signal'
    ))
    
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals['Close'],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Sell Signal'
    ))

    # Aggiungi indicatori per le candele engulfing
    for item in engulfing_candle:
        candle_index = item[0]
        direction = item[1]
        if direction == 'UP':
            marker_symbol = 'triangle-up'
            marker_color = 'green'
        elif direction == 'DOWN':
            marker_symbol = 'triangle-down'
            marker_color = 'red'
        fig.add_trace(go.Scatter(
            x=[candle_index],
            y=[df.loc[candle_index, 'High'] if direction == 'DOWN' else df.loc[candle_index, 'Low']],
            mode='markers',
            marker=dict(symbol=marker_symbol, color=marker_color, size=10),
            name=f'Engulfing {direction}'
        ))
    
    fig.update_layout(title='Candlestick chart with EMA, Bollinger Bands, and Trading Signals',
                      yaxis_title='Price',
                      xaxis_title='Date')
    fig.show()
def main():
    parser = argparse.ArgumentParser(description="Trading script")
    parser.add_argument('-b', '--brake', type=int, help="Calculate structure break index for the last N bars")
    parser.add_argument('-e', '--engulfing', type=int, default=0, help="Calculate engulfing candles for the last N bars")
    parser.add_argument('-c', '--chart', action='store_true', help="Display candlestick chart")
    args = parser.parse_args()

    os.system("clear")
    num_bars = args.brake if args.brake else 0
    num_engulfing_bars = args.engulfing if args.engulfing else 0
    df = init(args.brake is not None, num_bars, num_engulfing_bars)
    os.system("clear")
    tradeInfoManual(df)

    if args.chart:
        plot_chart(df)

if __name__ == "__main__":
    main()

