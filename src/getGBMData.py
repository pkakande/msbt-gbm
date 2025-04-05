import requests
import pandas as pd
import os
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from urllib3.exceptions import IncompleteRead

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Define the API endpoint
endpoint = "https://api.gdc.cancer.gov/files"

# Define the parameters
params = {
    "filters": {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": ["TCGA-GBM"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "access",
                   "value": ["open"]
                }
            },
            {
                "op": "not",
                "content": {
                    "field": "data_type",
                    "value": ["Slide Image"]
                }
            }
        ]
    },
    "fields": "file_id,file_name,file_size,data_type",
    "size": "10000",  # Increase size to ensure all files are retrieved
    "pretty": "true"
}

# Create a session with retry mechanism
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Make the request with a timeout
logger.info("Making POST request to retrieve data")
try:
    response = session.post(endpoint, json=params, timeout=10)  # 10 seconds timeout
except requests.exceptions.Timeout:
    logger.error("Request timed out")
    print("Request timed out")
    response = None

# Check the response status
if response and response.status_code == 200:
    data = response.json()
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data['data']['hits'])
    logger.info("Data retrieved successfully")
    print(df.head())

    # Filter out Slide Image data
    df = df[df['data_type'] != 'Slide Image']

    # Group by data_type and calculate total size for each type
    grouped = df.groupby('data_type')['file_size'].sum()

    # Convert the sizes to gigabytes and print the results
    for data_type, size_bytes in grouped.items():
        size_gb = size_bytes / (1024 ** 3)
        logger.info(f"Total size of {data_type} data files: {size_gb:.2f} GB")

    # Define the base download directory
    base_download_dir = "../data/raw/all_open_gbm_data"

    # Create the base directory if it doesn't exist
    if not os.path.exists(base_download_dir):
        os.makedirs(base_download_dir)

    # Download the files and organize by data type
    for index, row in df.iterrows():
        file_id = row['file_id']
        file_name = row['file_name']
        data_type = row['data_type']
        download_url = f"https://api.gdc.cancer.gov/data/{file_id}"

        # Create a directory for the data type if it doesn't exist
        data_type_dir = os.path.join(base_download_dir, data_type)
        if not os.path.exists(data_type_dir):
            os.makedirs(data_type_dir)

        # Log the file details before downloading
        logger.info(f"Attempting to download {file_name} (ID: {file_id})")

        # Download the file with a timeout and handle IncompleteRead
        try:
            response = session.get(download_url, stream=True, timeout=10)  # 10 seconds timeout
            response.raise_for_status()
            with open(os.path.join(data_type_dir, file_name), 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    f.write(chunk)
            logger.info(f"Downloaded {file_name} to {data_type_dir}")
        except requests.exceptions.Timeout:
            logger.error(f"Download timed out for {file_name} (ID: {file_id})")
            print(f"Download timed out for {file_name} (ID: {file_id})")
        except IncompleteRead as e:
            logger.error(f"Incomplete read: {e.partial} bytes read, {e.expected} more expected for {file_name} (ID: {file_id})")
            print(f"Incomplete read: {e.partial} bytes read, {e.expected} more expected for {file_name} (ID: {file_id})")

    # Function to check if a file is downloaded
    def is_file_downloaded(file_path):
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    # Function to check the download status of all files
    def check_download_status(df, base_download_dir):
        incomplete_files = []
        incomplete_sizes = []

        for index, row in df.iterrows():
            file_name = row['file_name']
            file_id = row['file_id']
            data_type = row['data_type']
            file_size = row['file_size']
            file_path = os.path.join(base_download_dir, data_type, file_name)

            if not is_file_downloaded(file_path):
                incomplete_files.append((file_name, file_id))
                incomplete_sizes.append(file_size)

        total_files = len(df)
        incomplete_count = len(incomplete_files)
        incomplete_percentage = (incomplete_count / total_files) * 100

        return incomplete_files, incomplete_sizes, incomplete_percentage

    # Check the download status
    incomplete_files, incomplete_sizes, incomplete_percentage = check_download_status(df, base_download_dir)

    logger.info("Checking download status")
    print("Incomplete or not downloaded files:")
    for file_name, file_size in zip(incomplete_files, incomplete_sizes):
        size_gb = file_size / (1024 ** 3)
        print(f"{file_name[0]} (ID: {file_name[1]}) - {size_gb:.2f} GB")

    print(f"Percentage of files not downloaded: {incomplete_percentage:.2f}%")

    # Write incomplete or not downloaded files to a file
    with open("incomplete_files.txt", "w") as f:
        f.write("Incomplete or not downloaded files:\n")
        for file_name, file_id in incomplete_files:
            f.write(f"{file_name} (ID: {file_id})\n")
        f.write(f"\nPercentage of files not downloaded: {incomplete_percentage:.2f}%\n")

else:
    logger.error("Failed to retrieve data")
    print("Failed to retrieve data")

