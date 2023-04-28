"""
This is a boilerplate pipeline 'cleaning'
generated using Kedro 0.18.7
"""

import sys
sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')

import os
import json
import pickle
import geojson
import regex as re
import numpy as np
import pandas as pd
import pymongo as pm
import geopandas as gpd
from kedro.config import ConfigLoader
from kedro.framework.project import settings
from kedro.extras.datasets.pickle import PickleDataSet
from retrieve_global_parameters import retrieve_catalog_path
from mongo_db import retrieve_database_and_collections, take_empty_collections

# -------------------------------------------- verify_cleaning_already_done --------------------------------------------
def verify_cleaning_already_done(day):
    is_done = False
    
    file_path = f"data/02_intermediate/is_done_cleaning_{day}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            is_done = pickle.load(file)
    
    return is_done

# --------------------------------------------- filter_data_geographically ---------------------------------------------
def filter_data_geographically(db_name, day, polygon_vertex: list, trigger_from_ingestion = False, already_done = False):
    finished = False
    # retrieve database and collections
    db, collections = retrieve_database_and_collections(db_name, day, ["processed", "clean"])
    # retrieve collections
    key_list = list(collections.keys())
    processed_collection = collections[key_list[0]]
    clean_collection = collections[key_list[1]]
    
    if clean_collection.count_documents({}) != 0:
        finished = True
        return finished
        
    if trigger_from_ingestion and not already_done:
        polygon = geojson.Polygon([polygon_vertex])
        # create index
        processed_collection.create_index([("geometry", pm.GEOSPHERE)])
        # filter data
        cursor = processed_collection.find(
        {"geometry": {"$geoWithin": {"$geometry": polygon}}})
        clean_collection.insert_many(cursor)
        
        finished = True
    
    return finished
    

# ------------------------------------------ keep_common_road_segments_across_time ------------------------------------------
def keep_common_road_segments_across_time(db_name, day, trigger = False, already_done = False):
    finished = False
    if trigger and not already_done:
        # retrieve database and collections
        db, collections = retrieve_database_and_collections(db_name, day, ["raw", "clean"])
        key_list = list(collections.keys())
        # retrieve collections
        raw_collection = collections[key_list[0]]
        clean_collection = collections[key_list[1]]
        # count distinct geometries
        distinct_geometry_values = clean_collection.distinct("geometry")
        geometries_count = pd.Series([clean_collection.count_documents({"geometry": geometry}) for geometry in distinct_geometry_values])
        # count raw documents
        n = raw_collection.count_documents({})
        # keep only geometries that appear n or 2n times
        gt_n_index = geometries_count\
        .where(geometries_count%n == 0)\
        .dropna()\
        .index
        geometry_to_keep = [distinct_geometry_values[i] for i in gt_n_index]
        clean_collection.delete_many({ "geometry": {"$not": { "$in": geometry_to_keep }} })
        
        finished = True
        
    return finished
    
    
# --------------------------------------------- remove_unnecessary_field ---------------------------------------------
def remove_unnecessary_fields(db_name, day, trigger = False, already_done = False):
    finished = False
    
    if trigger and not already_done:
        # retrieve database and collections
        db, collections = retrieve_database_and_collections(db_name, day, ["clean"])
        key_list = list(collections.keys())
        
        clean_collection = collections[key_list[0]]
        clean_collection.update_many({'currentFlow.subSegments': {'$exists': True}}, {'$unset': {'currentFlow.subSegments' : ''}})

        finished = True
        
    return finished


def update_data_catalog_trigger(trigger, day):
    finished = False
    file_path = f"data/02_intermediate/is_done_cleaning_{day}.pkl"
    if trigger:
        catalog_path = retrieve_catalog_path()
        with open(catalog_path, "r+") as f:
            contents = f.read()
            result = re.search(fr"is_done_cleaning_{day}:", contents)
            if result is None:
                contents = "\n".join([contents, 
                                  "\n    ".join([f"is_done_cleaning_{day}:",
                                                 f"type: pickle.PickleDataSet",
                                                 f"filepath: {file_path}"])])
            f.seek(0)
            f.truncate()
            f.write(contents)
        
        finished = True
        
        bool_data = PickleDataSet(filepath=file_path)
        bool_data.save(finished)
        
    return finished