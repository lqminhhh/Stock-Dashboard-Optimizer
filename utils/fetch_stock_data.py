# ============================================================
# fetch_stock_data.py
#
# Python script to fetch data from Yahoo Finance and store the
# data locally.
# ============================================================

# import packages
import yfinance as yf
from datetime import datetime, timedelta
import os

def fetch_stock_data(tickers, start_date, end_date):
    """
    fetch_stock_data:
    - Parameters:
        + tickers: lists of stocks
        + start_date: the start date of the data
        + end_date: the end date of the data
    - Output: the stocks data given the start and end date for the list of stocks
    """

    print(f"Fetching data for: {tickers} ...")

    try: 
        # Specify the file name
        filename = f"../data/stock_data_{start_date}_to_{end_date}.csv"

        # Check if file exists
        if os.path.exists(filename):
            print(f"{filename} already exists. Skipping save.")
            return filename

        # Check if input is valid
        if not tickers or not isinstance(tickers, list):
            raise ValueError("Ticker list must be a non-empty list of symbols.")
        
        data = yf.download(tickers, start = start_date, end = end_date)

        # Check if data is successfully downloaded
        if data.empty:
            raise ValueError("No data returned. Please check the input again!")
        
        # Transform/Clean data
        data = data.stack(level = 0).rename_axis(['Date', 'Ticker']).reset_index()
        data = data.rename(columns={'Ticker':'Type'})

        # Save file locally
        data.to_csv(filename, index = False)

        print(f"Sucessfully saved data to {filename}")
        return filename
    
    except ValueError as er:
        print(f"Value Error: {er}")
    except Exception as e:
        print(f"Error occured during fetch/save: {e}")

    return None

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'AMZN']  # List of companies
    end_date = datetime.today() # Today’s date
    start_date = end_date - timedelta(days = 30 * 5)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    fetch_stock_data(["AAPL", "MSFT", "GOOGL"], start_date, end_date)