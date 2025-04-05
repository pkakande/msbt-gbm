import requests
import pandas as pd
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

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

# Make the request
response = session.post(endpoint, json=params)

# Check the response status
if response.status_code == 200:
    data = response.json()
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data['data']['hits'])
    print(df.head())

    # Filter out Slide Image data
    df = df[df['data_type'] != 'Slide Image']

    # Group by data_type and calculate total size for each type
    grouped = df.groupby('data_type')['file_size'].sum()

    # Convert the sizes to gigabytes and print the results
    for data_type, size_bytes in grouped.items():
        size_gb = size_bytes / (1024 ** 3)
        print(f"Total size of {data_type} data files: {size_gb:.2f} GB")

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

        # Download the file
        response = session.get(download_url, stream=True)
        if response.status_code == 200:
            with open(os.path.join(data_type_dir, file_name), 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    f.write(chunk)
            print(f"Downloaded {file_name} to {data_type_dir}")
        else:
            print(f"Failed to download {file_name}")
else:
    print("Failed to retrieve data")

