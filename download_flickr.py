# download_flickr.py
import json
import os
import pandas as pd
import requests
from tqdm import tqdm


def download_and_prepare_flickr():
    """Download and prepare Flickr dataset"""

    # Download parquet URLs
    url = "https://huggingface.co/api/datasets/bigdata-pw/Flickr/parquet"
    output_file = "./data/parquet_urls.json"

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Saved parquet URLs to: {output_file}")

    # Download parquet files
    output_dir = "./Flickr/parquet/"
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "r") as f:
        fli = json.load(f)

    for part_url in tqdm(fli[:10], desc="Downloading parquets"):  # Download first 10 for demo
        file_name = part_url.split("/")[-1]
        output_path = os.path.join(output_dir, file_name)

        response = requests.get(part_url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    # Download images from parquet files
    save_root = './Flickr/train/'
    for parquet_file in os.listdir(output_dir):
        if parquet_file.endswith('.parquet'):
            parquet_path = os.path.join(output_dir, parquet_file)
            df = pd.read_parquet(parquet_path)

            output_image_dir = save_root + parquet_file.split(".")[0]
            os.makedirs(output_image_dir, exist_ok=True)

            for index, row in tqdm(df.iterrows(), desc=f"Processing {parquet_file}"):
                image_url = row.get('url_m')  # Medium size images
                image_id = row.get('id')

                if pd.notna(image_url):
                    try:
                        response = requests.get(image_url, stream=True, timeout=5)
                        if response.status_code == 200:
                            file_extension = image_url.split('.')[-1]
                            file_path = os.path.join(output_image_dir, f"{image_id}.{file_extension}")
                            with open(file_path, "wb") as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                    except Exception as e:
                        print(f"Error downloading {image_url}: {e}")


if __name__ == "__main__":
    download_and_prepare_flickr()