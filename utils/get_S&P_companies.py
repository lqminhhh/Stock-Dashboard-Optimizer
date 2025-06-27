# ============================================================
# get_S&P_companies.py
#
# Python script to get the list of S&P500 companies from 
# Wikipedia and save it as a csv file
# ============================================================

import pandas as pd 
import os

def extract_companies():
    """
    fetch_stock_data:
    - Parameters: None
    - Output: Save a list of S&P500 companies as a csv file
    """

    filename = '../data/S&P500.csv'

    # Check if file exists
    if not os.path.exists(filename):
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        df.to_csv(filename)
        print(f"Successfully save S&P500 data into {filename}")
        return None
    
    print("File already exist. Skip downloading data!")
    return None

if __name__ == "__main__":
    extract_companies()