import yfinance as yf
import pandas as pd 

def fetch_stock_data(tickers, period = "1d", interval = "1m"):
    df = yf.download(tickers, period = period, interval=interval)
    return df