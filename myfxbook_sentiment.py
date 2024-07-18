import requests
import re
from bs4 import BeautifulSoup


def getSentiment(symbol):
    page = requests.get("https://www.myfxbook.com/community/outlook/"+symbol)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="currentMetricsTable")
    out = results.text.replace("\n", " ").replace("\t", " ").replace("%", '').replace("lots", " ").strip()
    out = re.sub(r' +', ' ', out)
    out = out.split(" ")
    return (out[6:-4], out[10:])

