# ============================================================
# fetch_stock_data.py
#
# Python script to fetch data from Yahoo Finance and load the
# data to AWS S3 bucket.
# ============================================================

# import packages
import yfinance as yf
from datetime import datetime, timedelta
import os
import boto3
from botocore.exceptions import ClientError
import json
import pandas as pd

def load_aws_credentials():
    """
    load_aws_credentials:
    - Parameter: None
    - Output: Return acess key and secret key for AWS S3 user
    """
    try:
        with open("../private/config.json") as f:
            keys = json.load(f)

        return keys['aws_access_key_id'], keys['aws_secret_access_key']
    except Exception as e:
        print(f'Error loading AWS credentials: {e}. \
              Please check your credentials under private folder!')

def check_s3_bucket_and_object(access_key, secret_key, bucket_name, s3_key = None):
    """
    check_s3_bucket:
    - Parameter:
        + acess_key: Access key for AWS connection
        + secrete_key: Secrete key for AWS connection
        + bucket_name: Bucket name in S3
        + s3_ket: The object key stored in S3
    - Output: T/F if bucket/object exists in S3
    """
    
    try:
        s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        s3.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' exists.")
        object_exists = None

        # Check if object exists or not
        if s3_key:
            try:
                s3.head_object(Bucket = bucket_name, Key = s3_key)
                print(f"Object {s3_key} exists in bucket.")
                object_exists = True

            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    print(f"Object '{s3_key}' does NOT exist.")
                    object_exists = False
                else:
                    raise e
        return True, object_exists
    
    except boto3.exceptions.Boto3Error as e:
        print(f"Could not access bucket '{bucket_name}': {e}")
        return False, None


def create_s3_bucket(access_key, secret_key, bucket_name, region='us-east-2'):
    """
    create_s3_bucket:
    - Parameter:
        + acess_key: Access key for AWS connection
        + secrete_key: Secrete key for AWS connection
        + bucket_name: Bucket name in S3
    - Output: T/F if bucket successfully created
    """

    try:

        s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
        # Ohio region
        if region == 'us-east-2':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': region})
        
        print(f"Bucket '{bucket_name}' created successfully in region '{region}'.")
        return True
    except boto3.exceptions.Boto3Error as e:
        print(f"Failed to create bucket: {e}")
        return False


def upload_file_to_s3(access_key, secret_key, local_file_path, bucket_name, s3_key):
    """
    upload_file_to_s3:
    - Parameter:
        + acess_key: Access key for AWS connection
        + secrete_key: Secrete key for AWS connection
        + local_file_path: Local path where data is stored
        + bucket_name: Bucket name where data will be stored in S3
        + s3_key: object path inside the S3 bucket
    - Output: None. File with be uploaded to S3 bucket for further usage
    """

    try:
        s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        s3.upload_file(local_file_path, bucket_name, s3_key)
        print(f"Uploaded '{local_file_path}' to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Failed to upload to S3: {e}")

def clean_data(df):
    """
    clean_data:
    - Parameter:
        + df: DataFrame for cleaning
    - Output: Return cleaned version of data
    """

    # Transform/Clean data
    df = df.stack(level = 0).rename_axis(['Date', 'Ticker']).reset_index()
    df = df.rename(columns={'Ticker':'Type'})

    # Reformat the data
    df = df.melt(
        id_vars   = ["Date", "Type"],   
        var_name  = "Ticker",           
        value_name= "Price"
    )
    return df

def get_sp500_tickers(filename = '../data/S&P500.csv'):
    """
    get_sp500_tickers:
    - Parameters:
        + filename: data that has all S&P 500 tickers
    - Output: Return the list of all tickers
    """

    try:
        df = pd.read_csv(filename)

        # Ensure 'Symbol' column exists
        if 'Symbol' not in df.columns:
            raise ValueError("CSV file must contain a 'Symbol' column.")
        
        # Drop missing or blank values
        tickers = df['Symbol'].dropna().astype(str).str.strip()
        tickers = tickers[tickers != ''].tolist()

        # Replace '.' with '-' for Yahoo Finance format
        cleaned_tickers = [ticker.replace('.', '-') for ticker in tickers]

        # Final sanity check: no empty strings
        cleaned_tickers = [t for t in cleaned_tickers if t]


        return cleaned_tickers
    except Exception as e:
        print(f"Error in get_sp500_tickers: {e}")
        return []  
    

def fetch_stock_data(tickers, start_date, end_date, access_key, secret_key, bucket_name):
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
        # if os.path.exists(filename):
        #     print(f"{filename} already exists. Skipping save.")
        #     return filename

        # Check if input is valid
        if not tickers or not isinstance(tickers, list):
            raise ValueError("Ticker list must be a non-empty list of symbols.")
        
        data = yf.download(tickers, start = start_date, end = end_date)

        # Check if data is successfully downloaded
        if data.empty:
            raise ValueError("No data returned. Please check the input again!")
        
        # Clean & Reformat data
        data = clean_data(data)

        # Save file locally
        data.to_csv(filename, index = False)
        print(f"Sucessfully saved data to {filename}")

        # Check if bucket name is specified
        if bucket_name:
            # Generate s3 key
            s3_key = f'snapshots/{os.path.basename(filename)}'

            # Check if bucket and object exist
            bucket_ok, object_exists = check_s3_bucket_and_object(access_key, secret_key, bucket_name, s3_key)

            # If bucket doesn't exist, create it
            if not bucket_ok:
                created = create_s3_bucket(access_key, secret_key, bucket_name)
                if not created:
                    print("Aborting upload to S3!")
                    return filename

            # If object exists, skip upload
            if object_exists:
                print(f"Object '{s3_key}' already exists in S3. Skipping upload.")
            else:
                upload_file_to_s3(access_key, secret_key, filename, bucket_name, s3_key)
                print("Data successfully uploaded to S3!")

        return filename
    
    except ValueError as er:
        print(f"Value Error: {er}")
    except Exception as e:
        print(f"Error occured during fetch/save: {e}")

    return None

if __name__ == "__main__":
    # Load credentials for AWS S3
    access_key, secret_key = load_aws_credentials()

    # Get all the S&P 500 tickers
    tickers = get_sp500_tickers()

    # Define the start and end date
    end_date = datetime.today() # Today’s date
    start_date = datetime(2023, 1, 1)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # Specify the bucket name in AWS S3
    bucket_name = 'yfinancestockdata'

    # Download and save data in S3
    fetch_stock_data(tickers, start_date, end_date, access_key, secret_key, bucket_name)