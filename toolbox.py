import yfinance as yf
import pandas_ta as ta
import os
from tqdm import tqdm
import argparse

tqdm.pandas()

max_val = 0
min_val = 99999
structure_brake_index = []
engulfing_candle = []

def init(calculate_structure_brake, num_bars, num_engulfing_bars):
    #df = yf.Ticker("AAPL").history(period="1d", interval="5m")
    df = yf.Ticker("JPY=X").history(period="1d", interval="5m")
    df.index = df.index.tz_convert('Europe/Rome')
    df = df.dropna()

    df["EMA"] = ta.ema(df.Close, length=200)
    df["RSI"] = ta.rsi(df.Close, length=15)
    df["ATR"] = ta.atr(df.High, df.Low, df.Close, length=10)
    df["Trend"] = df.progress_apply(lambda row: ema_signal(df, row.name, 10), axis=1)
    
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
    print("EMA: ", df["Trend"].iloc[-1])
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

def main():
    parser = argparse.ArgumentParser(description="Trading script")
    parser.add_argument('-b', '--brake', type=int, help="Calculate structure break index for the last N bars")
    parser.add_argument('-e', '--engulfing', type=int, default=0, help="Calculate engulfing candles for the last N bars")
    args = parser.parse_args()

    os.system("clear")
    num_bars = args.brake if args.brake else 0
    num_engulfing_bars = args.engulfing if args.engulfing else 0
    df = init(args.brake is not None, num_bars, num_engulfing_bars)
    os.system("clear")
    tradeInfoManual(df)

if __name__ == "__main__":
    main()

