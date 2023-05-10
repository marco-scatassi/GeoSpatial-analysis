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
import regex as re
import numpy as np
import pandas as pd
from dateutil import parser
from pymongo import database
from get_api_call_time import get_api_call_time
from geojson import GeometryCollection, LineString
from kedro.extras.datasets.json import JSONDataSet
from kedro.extras.datasets.pickle import PickleDataSet
from mongo_db import retrieve_database_and_collections, take_empty_collections
from retrieve_global_parameters import retrieve_catalog_path, retrieve_db_name, retrieve_raw_data_root_dir

# -------------------------------------------- compose_url_to_raw_data ---------------------------------------------- #
def compose_url_to_raw_data(day: str):
    root_dir = retrieve_raw_data_root_dir()
    dirs = [dir for dir in os.listdir(root_dir+f"\\{day}") if dir[:10] in day]
    dirs_urls = [os.path.join(root_dir+f"\\{day}", dir) for dir in dirs]
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
        # create the geometry_length field
        geo_processed_data[-1]["geometry_length"] = [i['length'] for i in raw_data_links]
        # bring embedded fields to the top level
        for k in geo_processed_data[-1]['location']:
            geo_processed_data[-1][k] = geo_processed_data[-1]['location'][k]
            
        # remove duplicated fields    
        geo_processed_data[-1].pop('location')
        geo_processed_data[-1].pop('shape')
        
    return geo_processed_data


def process_raw_data(urls: list, day: str):
    db_name = retrieve_db_name()
    # retrieve database and collections wrappers
    db, collections = retrieve_database_and_collections(db_name, day, ["processed"])
    key_list = list(collections.keys())
    processed_collection = collections[key_list[0]]
    
    if processed_collection.count_documents({}) != 0:
        return []
    
    # load the raw data
    raw_data =  load_raw_data(urls)
    time_processed_collections_documents = splitting_and_time_processing(raw_data)
    processed_collections_documents = geometry_processing(time_processed_collections_documents)
    return processed_collections_documents

# ---------------------------------------------- insert_documents_in_the_collections -------------------------------- #
def insert_raw_data(urls: list, day: str):
    db_name = retrieve_db_name()
    # retrieve database and collections wrappers
    db, collections = retrieve_database_and_collections(db_name, day, ["raw"])
    key_list = list(collections.keys())
    raw_collection = collections[key_list[0]]
    
    # check if the collection is empty
    if raw_collection.count_documents({}) == 0:
        # load the raw data
        raw_data = load_raw_data(urls)
        # insert the documents in the collections
        raw_collection.insert_many(list(raw_data.values()))
        

def insert_processed_data(processed_data: dict, day: str):
    finished = False
    db_name = retrieve_db_name()
    # retrieve database and collections wrappers
    db, collections = retrieve_database_and_collections(db_name, day, ["raw", "processed"])
    empty_collections = take_empty_collections(collections)
    
    if len(empty_collections) == 0:
        finished = True
    
    # insert the documents in the collections
    for key, value in empty_collections.items():
        if "raw" in key:
            pass
        elif "processed" in key:
            value.insert_many(list(processed_data))
            finished = True
    
    return finished

# ---------------------------------------------- update_data_catalog_trigger --------------------------------------- #
def update_data_catalog_trigger(trigger, day):
    finished = False
    file_path = f"data/02_intermediate/cleaning_{day}_trigger_cleaning_{day}.pkl"
    if trigger:
        catalog_path = retrieve_catalog_path()
        with open(catalog_path, "r+") as f:
            contents = f.read()
            result = re.search(fr"cleaning.{day}.trigger_cleaning_{day}:", contents)
            if result is None:
                contents = "\n".join([contents, 
                                  "\n    ".join([f"cleaning.{day}.trigger_cleaning_{day}:",
                                                 f"type: pickle.PickleDataSet",
                                                 f"filepath: {file_path}"])])
                
            f.seek(0)
            f.truncate()
            f.write(contents)
        
        finished = True
        
        bool_data = PickleDataSet(filepath=file_path)
        bool_data.save(finished)
        
        
    return finished