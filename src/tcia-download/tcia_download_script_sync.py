"""
Script that downloads all of the data from The Cancer Imaging 
Archive. Iterates over a csv containing 
the name of every dataset to download and requests the 
images from TCIA. Then unzips the images and places 
them in the correct directory for each modality. 

Optionally takes a command line argument to start the 
download at a specific dataset, instead of 
downloading all of them.
"""
from tciaclient import TCIAClient
import urllib.request, urllib.error, urllib.parse, urllib.request, urllib.parse, urllib.error,sys
import pandas as pd
import json
import numpy as np
import time
import asyncio
import zipfile
from zipfile import ZipFile
import os
import subprocess
import logging
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", 
    required=False,
    default = "",
    help="specific dataset to start from")
parser.add_argument("dataset", 
    metavar="d",
    help="one of train, validate or test")
parser.add_argument("--out", 
    required=True,
    help="directory to save data in")
args = parser.parse_args()

output_dir = args.out
dataset = args.dataset
logging.basicConfig(filename=f'{dataset}.log', filemode="w", level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

modalities = ["MR", "CT", "PT", "CR", "DX","MG"]
limits = {key: 1e9 for key in modalities}

# for first download
# limits["MR"] = 1e2
# limits["CT"] = 1e2
# limits["PT"] = 1e2
# limits["MG"] = 1e2

#for final download
limits["MR"] = 1e5
limits["CT"] = 5e4
limits["PT"] = 1e9
limits["MG"] = 1e9

processed_collection_data = pd.read_csv(f"metadata/processed/{dataset}.csv")

start_index = 0 
if args.start:
    print(f"Start at {args.start}")
    collections = processed_collection_data["Collection"]

    start_index = list(collections).index(args.start)

tcia_client = TCIAClient(baseUrl="https://services.cancerimagingarchive.net/services/v4",resource = "TCIA")
start = time.time()
bytes_downloaded = 0
for collection_index in range(start_index, len(processed_collection_data)):
    collection = processed_collection_data["Collection"][collection_index]
    print(collection)
    logging.info(f"Downloading {collection}")
    for modality in modalities:
        image_count = 0

        collection = processed_collection_data["Collection"][collection_index]
        collection_data = pd.read_csv(f"metadata/{collection}.csv")
        
        for i in range(0, len(collection_data)):
            print(i, end="\r")
            if modality not in collection_data["Modality"][i]:
                continue
            series_uid = collection_data["SeriesInstanceUID"][i]
            
            tcia_client.get_image(series_uid, f"{output_dir}", f"temp-{collection}-{modality}-{i}.zip")
            try:
                with ZipFile(f"{output_dir}/temp-{collection}-{modality}-{i}.zip", "r") as zip_file:
                    zip_file.extractall(f"{output_dir}/{dataset}/{modality}/{collection}")
                os.remove(f"{output_dir}/temp-{collection}-{modality}-{i}.zip")
            except zipfile.BadZipFile:
                logging.warning(f"Bad zip file: {collection} --- {series_uid}")
            except IOError:
                logging.warning(f"File not found, {collection} --- {series_uid}")

            image_count += collection_data["ImageCount"][i]
            bytes_downloaded += collection_data["TotalSizeInBytes"][i]
            if image_count >= limits[modality]:
                print(f"Downloaded {image_count} images")
                break
    bot.email_me(message=f"{dataset} - Downloaded {collection}")

end = time.time()

print(f"Downloaded {bytes_downloaded // 1000000000}GB in {end-start} seconds")



