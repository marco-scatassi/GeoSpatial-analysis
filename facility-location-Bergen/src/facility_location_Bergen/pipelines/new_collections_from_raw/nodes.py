"""
This is a boilerplate pipeline 'new_collections_from_raw'
generated using Kedro 0.18.7
"""

import sys
sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')


# Get the database using the method we defined in pymongo_test_insert file
import os
import pytz
import copy
import json
import geojson
import numpy as np
import pandas as pd
from dateutil import parser
from pymongo import database
from get_api_call_time import get_api_call_time
from geojson import GeometryCollection, LineString
from kedro.extras.datasets.json import JSONDataSet
from mongo_db import retrieve_database_and_collections, take_empty_collections

# -------------------------------------------- compose_url_to_raw_data ---------------------------------------------- #
def compose_url_to_raw_data(db_name: str, day: str, root_dir: str):
    # retrieve database and collections wrappers
    db, collections = retrieve_database_and_collections(db_name, day, ["raw", "processed"])
    empty_collections = take_empty_collections(collections)
    # check if the number of empty collections is not more than 2
    if len(empty_collections) > 2:
        raise ValueError("No more than one date can be processed at a time, please select only one new date.")
    # compose the urls to the raw data
    day_ = np.unique([key[-10:] for key in empty_collections.keys()])
    # check if the day_ is not empty
    if len(day_) == 0:
        return []
    dirs = [dir for dir in os.listdir(root_dir+f"\\{day_[0]}") if dir[:10] in day_]
    dirs_urls = [os.path.join(root_dir+f"\\{day_[0]}", dir) for dir in dirs]
    file_urls = [os.path.join(dir_url, file) for dir_url in dirs_urls for file in os.listdir(dir_url)]
    return file_urls

# ----------------------------------------------- load_raw_data ---------------------------------------------------- #
def from_urls_to_JSONDataSet(urls: list):
    JSONDataSets = []
    for url in urls:
        JSONDataSets.append(JSONDataSet(filepath=url))
    return JSONDataSets


def load_raw_data(urls: list):
    raw_data = {}
    JSONDataSets = from_urls_to_JSONDataSet(urls)
    
    for url, json in zip(urls, JSONDataSets):
        if "afternoon" in url:
            key = url[-41:].removesuffix(".json")
        elif "midday" in url:
            key = url[-38:].removesuffix(".json")
        elif "morning" in url:
            key = url[-39:].removesuffix(".json")
            
        raw_data[key] = json.load()
    
    return raw_data

# ---------------------------------------------- process_raw_data -------------------------------------------------- #
def get_time_from_raw_data(raw_data: dict):
    times = {}
    for key, value in raw_data.items():
        times[key] = get_api_call_time(key)
    return times


def splitting_and_time_processing(raw_data: dict):
    time_processed_collections_documents = []
    api_call_times = get_time_from_raw_data(raw_data)
    
    for key, value in raw_data.items():
        dt = parser.parse(value["sourceUpdated"])
        
        for result in value["results"]:
            time_processed_collections_documents.append(result)
            time_processed_collections_documents[-1]["sourceUpdated"] = dt  
            time_processed_collections_documents[-1]["api_call_time"] = api_call_times[key] 
        
    return time_processed_collections_documents


def geometry_processing(input_data: list):
    geo_processed_data = []
    
    for doc in input_data:
        geo_processed_data.append(doc)
        # extract the links field from the input data
        raw_data_links = doc['location']['shape']['links']
        # create the geometry field (in order to comply the geojson format)
        geo_processed_data[-1]["geometry"] = GeometryCollection(
            [LineString([(e['lng'],e['lat']) for e in i['points']])for i in raw_data_links])
            
        # bring embedded fields to the top level
        for k in geo_processed_data[-1]['location']:
            geo_processed_data[-1][k] = geo_processed_data[-1]['location'][k]
            
        # remove duplicated fields    
        geo_processed_data[-1].pop('location')
        geo_processed_data[-1].pop('shape')
        
    return geo_processed_data


def process_raw_data(urls: list):
    # load the raw data
    raw_data =  load_raw_data(urls)
    time_processed_collections_documents = splitting_and_time_processing(raw_data)
    processed_collections_documents = geometry_processing(time_processed_collections_documents)
    return processed_collections_documents

# ---------------------------------------------- insert_documents_in_the_collections -------------------------------- #
def insert_raw_data(urls: list, db_name: str, day: str):
    # retrieve database and collections wrappers
    db, collections = retrieve_database_and_collections(db_name, day, ["raw", "processed"])
    empty_collections = take_empty_collections(collections)
    
    # load the raw data
    raw_data = load_raw_data(urls)
    
    # insert the documents in the collections
    for key, value in empty_collections.items():
        if "raw" in key:
            value.insert_many(list(raw_data.values()))
        else:
            pass

def insert_processed_data(processed_data: dict, db_name: str, day: str):
    # retrieve database and collections wrappers
    db, collections = retrieve_database_and_collections(db_name, day, ["raw", "processed"])
    empty_collections = take_empty_collections(collections)
    
    # insert the documents in the collections
    for key, value in empty_collections.items():
        if "raw" in key:
            pass
        elif "processed" in key:
            value.insert_many(list(processed_data))
