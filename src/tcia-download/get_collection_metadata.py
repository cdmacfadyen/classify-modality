"""Script to download the metadata for every 
TCIA dataset. The metadata collected 
at this stage is the Modality, SeriesInstanceUID,  ImageCount, 
BodyPartExamined, SeriesDescription, PatientID. These 
are useful things to have going forward. 
"""

from tciaclient import TCIAClient
import urllib.request, urllib.error, urllib.parse, urllib.request, urllib.parse, urllib.error,sys
import pandas as pd
import json
import numpy as np
from io import StringIO

def printServerResponse(response):
    if response.getcode() == 200:
        print("Server Returned:\n")
        print(response.read().decode())
        print("\n")
    else:
        print("Error: " + str(response.getcode()))


def get_series_sizes(metadata, tcia_client):
    size_list = []
    for uid in metadata["SeriesInstanceUID"]:
        print(uid, end="\r")
        series_size_response = tcia_client.get_series_size(uid)
        size_dict = json.loads(series_size_response.read().decode())
        size = int(float(size_dict[0]["TotalSizeInBytes"])) # cast it to a float first because it has a . in it, then cast to an int because you aren't going to have half a byte
        size_list.append(size)
    return size_list

tcia_client = TCIAClient(baseUrl="https://services.cancerimagingarchive.net/services/v4",resource = "TCIA")

links_csv_path = "./sorted-api-links.csv" 

links_csv = pd.read_csv(links_csv_path)
collections = links_csv["Collection"]


collections = list(collections)
number_of_collections = len(collections)
counter = 0
while len(collections) > 0:
    try:
        counter = (counter + 1) % len(collections)
        collection = collections[counter] 
        print(collection)
        print(counter, len(collections))
        series_response = tcia_client.get_series(collection = collection, outputFormat="json")
        response_as_string = series_response.read().decode()
        response_df = pd.read_json(StringIO(response_as_string))

        columns_to_take = ["Modality", "SeriesInstanceUID", "ImageCount", "BodyPartExamined", "SeriesDescription", "PatientID"]
        metadata = pd.DataFrame()
        column_length = len(response_df["Modality"])
        for column in columns_to_take: 
            try:
                metadata[column] = response_df[column]
            except KeyError:
                metadata[column] = [np.NaN] *  column_length    # makes a list `column_length` long of NaN
        
        series_sizes = get_series_sizes(metadata, tcia_client)
        metadata["TotalSizeInBytes"] = series_sizes
        metadata.to_csv(f"./metadata/{collection}.csv")
        collections[:] = [c for c in collections if c is not collection]
    except KeyboardInterrupt as e:
        raise
    except Exception as e:
        print(e)

