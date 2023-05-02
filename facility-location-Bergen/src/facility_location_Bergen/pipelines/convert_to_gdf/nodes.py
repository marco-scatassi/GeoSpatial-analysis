import sys
sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')

import os
import regex as re
import pandas as pd
import geopandas as gpd
from datetime import datetime
from mongo_db import retrieve_database_and_collections
from kedro.extras.datasets.pickle import PickleDataSet
from retrieve_global_parameters import retrieve_catalog_path
from convert_geometry import toMultiLineString, toExtremePoints
from log import print_INFO_message_timestamp, print_INFO_message
from retrieve_file_path import retrieve_gdf_path, retrieve_gif_saving_path


# -------------------------------------------- verify_gdf_already_created --------------------------------------------
def verify_gdf_already_created(date: dict):
    is_created = False
    # verify if the file already exists
    saving_path = retrieve_gdf_path(date)
    if os.path.exists(saving_path):
        is_created = True
    
    return is_created


# -------------------------------- retrieve_data_from_mongoDB  and convert them to a geopandas df --------------------------------
def from_json_to_gdf(db_name:str, date: dict, already_created: bool, saving_path: str):
    finished = already_created
    if not already_created:
        # set the first and last date of the date
        first_date = datetime.strptime(f"{date['day']}T07:30:00.000+02:00", "%d_%m_%YT%H:%M:%S.%f%z")
        last_date = datetime.strptime(f"{date['day']}T17:00:00.000+02:00", "%d_%m_%YT%H:%M:%S.%f%z")
        # retrieve database and collections
        db, collection = retrieve_database_and_collections(db_name, date['day'], ["clean"])
        key_list = list(collection.keys())
        clean_collection = collection[key_list[0]]
        # retrieve data from mongoDB and convert it to a pandas dataframe
        df = pd.json_normalize(clean_collection.find({"api_call_time": {"$gte": first_date, "$lte": last_date}}))
        # process dataframe columns
        df["_id"] = df.apply(lambda x: str(x["_id"]), axis=1)
        df['geometry.multi_line'] = df.apply(lambda x: toMultiLineString(x["geometry.geometries"]), axis=1)
        df['geometry.extreme_point'] = df.apply(lambda x: toExtremePoints(x["geometry.multi_line"]), axis=1)
        df['api_call_time'] = df.apply(lambda x: x["api_call_time"].tz_localize('UTC'), axis=1)
        df['sourceUpdated'] = df.apply(lambda x: x["sourceUpdated"].tz_localize('UTC'), axis=1)
        # set the index of the dataframe
        df.set_index("_id", inplace=True)
        # drop unnecessary columns
        df.drop(columns=["geometry.geometries"], inplace=True)
        # convert dataframe to geodataframe
        gdf = gpd.GeoDataFrame(df, geometry="geometry.multi_line")
        # save geodataframe to a geojson file
        gdf_ = PickleDataSet(filepath=saving_path)
        gdf_.save(gdf)
        finished = True
        
    return finished


# ---------------------------------------------- update_data_catalog --------------------------------------- 
def update_data_catalog_gdf(date, trigger, gdf_path):
    finished = False
    if trigger:
        catalog_path = retrieve_catalog_path()
        
        with open(catalog_path, "r+") as f:
            contents = f.read()
            result = re.search(fr"gdf{date['day']}:", contents)
            if result is None:
                contents = "\n".join([contents, 
                                  "\n    ".join([f"gdf{date['day']}:",
                                                 f"type: pickle.PickleDataSet",
                                                 f"filepath: {gdf_path}"])])
                
            f.seek(0)
            f.truncate()
            f.write(contents)
        
        finished = True
        
    return finished 