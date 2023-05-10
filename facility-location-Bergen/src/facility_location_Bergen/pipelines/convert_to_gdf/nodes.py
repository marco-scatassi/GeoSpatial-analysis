import sys
sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')

import warnings
from shapely.errors import ShapelyDeprecationWarning
# Ignore the ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
import copy
import numpy as np
import regex as re
import pandas as pd
import datetime as dt
import geopandas as gpd
from datetime import datetime
from shapely.geometry import MultiLineString
from kedro.extras.datasets.pickle import PickleDataSet
from mongo_db import retrieve_database_and_collections
from convert_geometry import toMultiLineString, toExtremePoints
from log import print_INFO_message_timestamp, print_INFO_message
from retrieve_global_parameters import retrieve_catalog_path, retrieve_gdf_path, retrieve_db_name

################################################################################## STEP 1 ######################################################################

# -------------------------------------------- verify_gdf_already_created --------------------------------------------
def verify_gdf_already_created(date: dict):
    is_created = False
    # verify if the file already exists
    saving_path = retrieve_gdf_path(date)
    if os.path.exists(saving_path):
        is_created = True
    
    return is_created

################################################################################## STEP 2 ######################################################################

# -------------------------------- retrieve_data_from_mongoDB  and convert them to a geopandas df --------------------------------
def from_json_to_gdf(date: dict, already_created: bool):
    finished = already_created
    db_name = retrieve_db_name()
    saving_path = retrieve_gdf_path(date)
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

################################################################################## STEP 3 ######################################################################

# --------------------------------------- verify process_gdf already done ---------------------------------------
def verify_process_gdf_already_done(date: dict, trigger: bool):
    already_done = False
    if trigger:
        # verify if the file already exists
        saving_path = retrieve_gdf_path(date, processed=True)
        if os.path.exists(saving_path):
            already_done = True
    
    return already_done

################################################################################## STEP 4 ######################################################################

# ------------------------------------------------ get_length ------------------------------------------------
def get_length(data):
    n = len(data["currentFlow.subSegments"])
    length_list = []
    index = 0

    # print_INFO_message_timestamp("Start")

    for i in range(n):
        # print_INFO_message("Index: {}".format(index))
        length_list.append([])
        length = data["currentFlow.subSegments"][i]["length"]
        
        for j in range(index, len(data["geometry_length"])):
            # print(j, len(data["geometry_length"]))
            length_list[i].append(data["geometry_length"][j])
            rel_error = abs(length - sum(length_list[i]))/length
            
            if j < len(data["geometry_length"])-2:
                next = sum(length_list[i]) + data["geometry_length"][j+1]
                rel_error_next = abs(length - next)/length
                
                if rel_error_next > rel_error:
                    index = j+1
                    break
    
    # if the total length of the subsegments is not equal to the length of the road
    # we avoid to split the road in subsegments
    # the previous happens in two situatons:
    # 1) the approximation of the final segment get worse when adding the last length
    # 2) the approximation of an intermediate segments improves always when adding the next length
    
    if sum([len(x) for x in length_list]) != len(data["geometry_length"]):
        # print([len(x) for x in length_list])
        # print([len(length_list), n])
        # print([len(data["geometry_length"]), len(data["geometry.multi_line"])])
        return None
    
    total_difference = 0
    for i in range(n):
        total_difference += abs(data["currentFlow.subSegments"][i]["length"] - sum(length_list[i]))
        relative_difference = total_difference/data["length"]
    if relative_difference > 0.01:
        print_INFO_message("Relative difference: {}".format(relative_difference))
        return None
        
    return length_list


# ------------------------------------------------ get_geometry ------------------------------------------------
def get_geometry(data, length_list):
    n = len(data["currentFlow.subSegments"])
    geometry_list = []
    
    # print_INFO_message_timestamp("Start")
    
    counter = 0
    
    for i in range(n):
        # print_INFO_message("Index: {}".format(i))
        geometry_list.append([])
        for j in range(len(length_list[i])):
            geometry_list[i].append(data["geometry.multi_line"][counter])
            counter += 1
            
        geometry_list[i] = geometry_list[i]
    return geometry_list


# ------------------------------------------------ new_rows ------------------------------------------------ 
def new_rows(data, gdf):
    n = len(data["currentFlow.subSegments"])
    new_rows = [{k: np.nan for k in gdf.columns} for i in range(n)]
    length_list = get_length(data)
        
    if length_list is None:
        return None
        
    geometry_list = get_geometry(data, length_list)
        
    for i in range(n):
        new_rows[i]["geometry.multi_line"] = [geometry_list[i]]
        new_rows[i]["geometry.extreme_point"] = [toExtremePoints(MultiLineString(geometry_list[i]))]
        new_rows[i]["geometry_length"] = [length_list[i]]
        new_rows[i]["currentFlow.subSegments"] = np.nan
        new_rows[i]["currentFlow.traversability"] = data["currentFlow.subSegments"][i]["traversability"]
            
        try:
            new_rows[i]["currentFlow.junctionTraversability"] = data["currentFlow.subSegments"][i]["junctionTraversability"]
        except:
            new_rows[i]["currentFlow.junctionTraversability"] = np.nan
                
        try:
            new_rows[i]["currentFlow.jamTendency"] = data["currentFlow.subSegments"][i]["jamTendency"]
        except:
            new_rows[i]["currentFlow.jamTendency"] = np.nan
            
        try:
            new_rows[i]["currentFlow.confidence"] = data["currentFlow.subSegments"][i]["confidence"]
        except:
            new_rows[i]["currentFlow.confidence"] = np.nan
        
        try:
            new_rows[i]["currentFlow.speed"] = data["currentFlow.subSegments"][i]["speed"]
        except:
            new_rows[i]["currentFlow.speed"] = np.nan
        
        try: 
            new_rows[i]["currentFlow.speedUncapped"] = data["currentFlow.subSegments"][i]["speedUncapped"]
        except:
            new_rows[i]["currentFlow.speedUncapped"] = np.nan
            
        new_rows[i]["length"] = data["currentFlow.subSegments"][i]["length"]
        new_rows[i]["currentFlow.freeFlow"] = data["currentFlow.subSegments"][i]["freeFlow"]
        new_rows[i]["currentFlow.jamFactor"] = data["currentFlow.subSegments"][i]["jamFactor"]
        
        new_rows[i]["api_call_time"] = data["api_call_time"]
        new_rows[i]["sourceUpdated"] = data["sourceUpdated"]
        new_rows[i]["description"] = data["description"]     
        new_rows[i]["geometry.type"] = data["geometry.type"]
        
    df_list = []
        
    for i in range(n):
        df_list.append(pd.DataFrame(new_rows[i]))
        df_list[-1]["geometry.multi_line"] = df_list[-1]["geometry.multi_line"].apply(lambda x: MultiLineString(x))
    
    df = pd.concat(df_list, ignore_index=True)
    
    if len(df) != n:
        print("What the fuck")
    return df


# --------------------------------------- sub_segment_exctraction ---------------------------------------
def sub_segment_extraction(gdf):
    new_rows_ = []
    indexes = []

    print_INFO_message_timestamp("Start")

    for i in range(len(gdf)):
        data = gdf.iloc[i]
        try:
            isna = np.isnan(data["currentFlow.subSegments"])
        except:
            isna = False
        
        if isna == False:
            print_INFO_message("Index: {}".format(i))
            nr = new_rows(data, gdf)
            if nr is not None:
                new_rows_.append(nr)
                indexes.append(i)

    gdf_processed = gdf.reset_index(drop=True).drop(index = indexes)
    gdf_processed = pd.concat([gdf_processed] + new_rows_, ignore_index=True).reset_index(drop=True)
    
    if gdf_processed["currentFlow.subSegments"].isna().all():
        gdf_processed = gdf_processed[['sourceUpdated',
                                        'api_call_time',
                                        'geometry_length',
                                        'description',
                                        'length',
                                        'currentFlow.speed',
                                        'currentFlow.speedUncapped',
                                        'currentFlow.freeFlow',
                                        'currentFlow.jamFactor',
                                        'currentFlow.confidence',
                                        'currentFlow.traversability',
                                        'geometry.type',
                                        'currentFlow.jamTendency',
                                        'currentFlow.junctionTraversability',
                                        'geometry.multi_line',
                                        'geometry.extreme_point']]
    
    return gdf_processed

# --------------------------------------- process_closed_roads ---------------------------------------
def process_closed_roads(gdf):
    gdf_copy = copy.deepcopy(gdf)
    gdf_ = gdf_copy[gdf_copy["currentFlow.traversability"]=="closed"]
    indexes = gdf_.index
    for idx in indexes:
        gdf_copy.loc[idx, "currentFlow.speed"] = 0
        gdf_copy.loc[idx, "currentFlow.speedUncapped"] = 0
    
    return gdf_copy


# --------------------------------------- process_gdf ---------------------------------------
def process_gdf(date, already_done):
    finished = False
    
    if not already_done:
        # read gdf
        gdf_path = retrieve_gdf_path(date)
        gdf = pd.read_pickle(gdf_path)
        # process gdf
        gdf_ = sub_segment_extraction(gdf)
        gdf_ = process_closed_roads(gdf_)
        # save gdf
        saving_path = retrieve_gdf_path(date, processed=True)
        dataset = PickleDataSet(filepath=saving_path)
        dataset.save(gdf_)
            
        finished = True 
    else:
        finished = True
        
    return finished

################################################################################## STEP 5 ######################################################################

# ---------------------------------------------- update_data_catalog --------------------------------------- 
def update_data_catalog_gdf(date, trigger):
    finished = False
    if trigger:
        catalog_path = retrieve_catalog_path()
        gdf_path = retrieve_gdf_path(date)
        gdf_processed_path = retrieve_gdf_path(date, processed=True)
        
        with open(catalog_path, "r+") as f:
            contents = f.read()
            result = re.search(fr"gdf{date['day']}:", contents)
            if result is None:
                contents = "\n".join([contents, 
                                  "\n    ".join([f"convert_to_gdf.{date['day']}.gdf.{date['day']}:",
                                                 f"type: pickle.PickleDataSet",
                                                 f"filepath: {gdf_path}"])])
                
            f.seek(0)
            f.truncate()
            f.write(contents)
            
        with open(catalog_path, "r+") as f:
            contents = f.read()
            result = re.search(fr"gdf{date['day']}_processed:", contents)
            if result is None:
                contents = "\n".join([contents, 
                                  "\n    ".join([f"convert_to_gdf.{date['day']}.gdf_processed.{date['day']}:",
                                                 f"type: pickle.PickleDataSet",
                                                 f"filepath: {gdf_processed_path}"])])
                
            f.seek(0)
            f.truncate()
            f.write(contents)
        
        finished = True
        
    return finished