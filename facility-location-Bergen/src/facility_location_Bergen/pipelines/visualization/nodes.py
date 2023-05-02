import sys
sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')

import io
import os
import pytz
import regex as re
import numpy as np
import pandas as pd
from time import time
from PIL import Image
import geopandas as gpd
import cartopy.crs as ccrs
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt
from urllib.request import urlopen, Request
from matplotlib.animation import FuncAnimation
from mongo_db import retrieve_database_and_collections
from kedro.extras.datasets.pickle import PickleDataSet
from retrieve_global_parameters import retrieve_catalog_path
from convert_geometry import toMultiLineString, toExtremePoints
from log import print_INFO_message_timestamp, print_INFO_message

# ------------------------------------------------- define saving paths -------------------------------------------------
def define_saving_paths_gdf(date: dict):
    # define saving paths
    saving_path = f"data/03_primary/{date['day']}.geojson"
    return saving_path

def define_saving_paths_gif(date: dict):
    # define saving paths
    saving_path = f"data/08_reporting/AnimatedPlot{date['day']}{date['time']}.gif"
    return saving_path

# -------------------------------------------- verify_gdf_already_created --------------------------------------------
def verify_gdf_already_created(saving_path):
    is_created = False
    # verify if the file already exists
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

# -------------------------------------------- process data to plot them --------------------------------------------
# define the color of the lines based on the jam factor
def get_color(jam):
    if jam is np.nan:
        return np.nan
    
    if 0 <= jam <= 3:
        return "green"
    elif 3 < jam <= 7:
        return "orange"
    elif 7 <= jam <= 10:
        return "red"
    
# define the first and last date of the date based on the time of the date
def get_dates(date: str):
    if date['time']=="morning":
        first_date = datetime.strptime(f"{date['day']}T07:30:00.000+02:00", "%d_%m_%YT%H:%M:%S.%f%z")
        last_date = datetime.strptime(f"{date['day']}T09:30:00.000+02:00", "%d_%m_%YT%H:%M:%S.%f%z")
    elif date['time']=="midday":   
        first_date = datetime.strptime(f"{date['day']}T11:00:00.000+02:00", "%d_%m_%YT%H:%M:%S.%f%z")
        last_date = datetime.strptime(f"{date['day']}T12:00:00.000+02:00", "%d_%m_%YT%H:%M:%S.%f%z")
    elif date['time']=="afternoon":
        first_date = datetime.strptime(f"{date['day']}T15:00:00.000+02:00", "%d_%m_%YT%H:%M:%S.%f%z")
        last_date = datetime.strptime(f"{date['day']}T17:00:00.000+02:00", "%d_%m_%YT%H:%M:%S.%f%z")
    else:
        raise ValueError("time must be 'morning', 'midday' or 'afternoon'")
    
    return first_date, last_date

    
# process geodataframe to plot data
def process_gdf_to_plot_data(date: dict, file_path: str):
    print_INFO_message_timestamp(f"Processing data for {date['day']} {date['time']}")
    
    t = time()
    # initialize variables
    lats = {}
    lons = {}
    df_dict = {}
    first_date, last_date = get_dates(date)
    # load geodataframe
    gdf_ = PickleDataSet(filepath=file_path) 
    gdf = gdf_.load()
    gdf_sample = gdf.where(
        (gdf["api_call_time"]>=first_date) & (gdf["api_call_time"]<=last_date)
        ).dropna(how="all")

    print_INFO_message(f"Time to load the geodataframe: {time()-t:.2f} seconds")
    
    # process geodataframe
    for i, date in enumerate(gdf_sample["api_call_time"].unique()):
        df = gdf_sample.where(gdf_sample["api_call_time"]==date).dropna(how="all")
        df_dict[date]=[]
        
        for feature, date, jam in zip(df['geometry.multi_line'], df["api_call_time"], df["currentFlow.jamFactor"]):
            color = get_color(jam)
            linestrings = feature.geoms
            lats = []
            lons = []
            dates = []
            colors = []
            
            for linestring in linestrings:
                x, y = linestring.xy
                lats = np.append(lats, list(x))
                lons = np.append(lons, list(y))
                dates = np.append(dates, [date]*len(x))
                colors = np.append(colors, [color]*len(x))
            
            df_dict[date].append(pd.DataFrame({"lat": lats, "lon": lons, "date": dates, "color": colors}))
            df_dict[date][-1] = pd.concat([df_dict[date][-1], 
                                        pd.DataFrame({"lat": [np.nan], "lon": [np.nan], "date": [np.nan], "color": [np.nan]})],
                                        ignore_index=True)
            
        # print time
        if i%2==0:
            print_INFO_message(f"Time to process {i+1} of {len(gdf_sample['api_call_time'].unique())} dates: {time()-t:.2f} seconds")
    # concatenate dataframes
    df_dict2 = {}
    for key in df_dict.keys():
        df_dict2[key] = pd.concat(df_dict[key], ignore_index=True)
        
    return df_dict2

# -------------------------------------------- define figure canva --------------------------------------------
def image_spoof(self, tile): 
    url = self._image_url(tile) # get the url of the street map API
    req = Request(url) # start request
    req.add_header('User-agent','Anaconda 3') # add user agent to request
    fh = urlopen(req) 
    im_data = io.BytesIO(fh.read()) # get image
    fh.close() # close url
    img = Image.open(im_data) # open image with PIL
    img = img.convert(self.desired_tile_form) # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy

def set_figure_background():
    cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
    osm_img = cimgt.OSM() # spoofed, downloaded street map
    fig = plt.figure(figsize=(12,9)) # open matplotlib figure
    ax = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map

    center_pt = [60.39299, 5.32415] # lat/lon of One World Trade Center in NYC
    zoom = 0.15 # for zooming out of center point
    extent = [center_pt[1]-(zoom*1.3),center_pt[1]+(zoom*1.6),center_pt[0]-zoom,center_pt[0]+zoom] # adjust to zoom
    ax.set_extent(extent) # set extents

    scale = np.ceil(-np.sqrt(2)*np.log(np.divide(zoom,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    ax.add_image(osm_img, int(scale))

    ax.set_title('Traffic across time in Bergen')

    return fig, ax

# -------------------------------------------- create and save the animation --------------------------------------------
def animate(i, ax, df_dict2, lines=[]):
    if i == 100:
        df = df_dict2[list(df_dict2.keys())[0]] 
        ax.plot('lat', 'lon', data=df, linewidth=0, alpha=0, transform=ccrs.Geodetic())
    else:
        df = df_dict2[list(df_dict2.keys())[i]] 
        ts = time()
        
        ax.plot('lat', 'lon', data=df.where(df.color=="green"), linewidth=0.5, c="green", transform=ccrs.Geodetic())
        ax.plot('lat', 'lon', data=df.where(df.color=="orange"), linewidth=0.5, c="orange", transform=ccrs.Geodetic())
        ax.plot('lat', 'lon', data=df.where(df.color=="red"), linewidth=0.5, c="red", transform=ccrs.Geodetic())
        
        ax.set_title(f"Traffic across time in Bergen: {list(df_dict2.keys())[i]}")
        print_INFO_message(f"Time to plot the {i+1}th of {len(df_dict2.keys())} dates: {time()-ts:.2f} seconds")
    
def create_and_save_animation(date: dict, file_path: str, saving_path: str, trigger: bool):
    finished = False
    if trigger:
        df_dict2 = process_gdf_to_plot_data(date, file_path)
        fig, ax = set_figure_background()   
        
        ani = FuncAnimation(fig=fig, 
                        func=partial(animate, ax=ax, df_dict2=df_dict2),
                        frames=[100]+list(range(len(df_dict2.keys()))), 
                        interval=1000, 
                        repeat_delay=False,
                        cache_frame_data=False)
        
        ani.save(saving_path, dpi=200)
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

def update_data_catalog_gif(date, trigger, gif_path):
    finished = False
    if trigger:
        catalog_path = retrieve_catalog_path()
        
        with open(catalog_path, "r+") as f:
            contents = f.read()
            result = re.search(fr"visualization.gif{date['day']}{date['time']}:", contents)
            if result is None:
                contents = "\n".join([contents, 
                                  "\n    ".join([f"visualization.gif{date['day']}{date['time']}:",
                                                 f"type: matplotlib.MatplotlibWriter",
                                                 f"filepath: {gif_path}"])])
                
            f.seek(0)
            f.truncate()
            f.write(contents)
        
        finished = True
        
    return finished    
