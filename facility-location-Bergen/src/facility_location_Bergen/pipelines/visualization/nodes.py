import sys
sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')
import io
import pytz
import numpy as np
import pandas as pd
from time import time
from PIL import Image
import geopandas as gpd
import cartopy.crs as ccrs
from datetime import datetime
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from matplotlib.animation import FuncAnimation
from mongo_db import retrieve_database_and_collections
from convert_geometry import toMultiLineString, toExtremePoints
def from_json_to_gdf(db_name:str, day: str):
    # set the first and last date of the day
    first_date = datetime.strptime(f"{day}T07:30:00.000+02:00", "%d_%m_%YT%H:%M:%S.%f%z")
    last_date = datetime.strptime(f"{day}T17:00:00.000+02:00", "%d_%m_%YT%H:%M:%S.%f%z")
    # retrieve database and collections
    db, collection = retrieve_database_and_collections(db_name, day, ["clean"])
    key_list = list(collection.keys())
    clean_collection = collection[key_list[0]]
    # retrieve data from mongoDB and convert it to a pandas dataframe
    df = pd.json_normalize(clean_collection.find({"api_call_time": {"$gte": first_date, "$lte": last_date}}))
    df.set_index("_id", inplace=True)
    # process dataframe columns
    df['geometry.multi_line'] = df.apply(lambda x: toMultiLineString(x["geometry.geometries"]), axis=1)
    df['geometry.extreme_point'] = df.apply(lambda x: toExtremePoints(x["geometry.multi_line"]), axis=1)
    df['api_call_time'] = df.apply(lambda x: x["api_call_time"].tz_localize('UTC'), axis=1)
    df['sourceUpdated'] = df.apply(lambda x: x["sourceUpdated"].tz_localize('UTC'), axis=1)
    # convert dataframe to geodataframe
    gdf = gpd.GeoDataFrame(df, geometry="geometry.multi_line")
    return gdf
gdf_sample = gdf.where((gdf["api_call_time"]>=first_date) & (gdf["api_call_time"]<=last_date)).dropna(how="all")
def get_color(jam):
    if jam is np.nan:
        return np.nan
    
    if 0 <= jam <= 3:
        return "green"
    elif 3 < jam <= 7:
        return "orange"
    elif 7 <= jam <= 10:
        return "red"
lats = {}
lons = {}
df_dict = {}
t = time()

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
        
    if i%2==0:
        print(f"Time after {i} iter: {time()-t}")
df_dict2 = {}

for key in df_dict.keys():
    df_dict2[key] = pd.concat(df_dict[key], ignore_index=True)
def image_spoof(self, tile): # this function pretends not to be a Python script
    url = self._image_url(tile) # get the url of the street map API
    req = Request(url) # start request
    req.add_header('User-agent','Anaconda 3') # add user agent to request
    fh = urlopen(req) 
    im_data = io.BytesIO(fh.read()) # get image
    fh.close() # close url
    img = Image.open(im_data) # open image with PIL
    img = img.convert(self.desired_tile_form) # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy
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
# ax.set_xlabel('longitude')
# ax.set_ylabel('latitude')
# t = ax.text(5.4,60.25,'', fontdict={'family': 'serif',
#                 'color':  'white',
#                 'weight': 'normal',
#                 'size': 13,
#                 })
def animate(i):
    df = df_dict2[list(df_dict2.keys())[i]] 
    ts = time()
    
    ax.plot('lat', 'lon', data=df.where(df.color=="green"), c="green", transform=ccrs.Geodetic())
    ax.plot('lat', 'lon', data=df.where(df.color=="orange"), c="orange", transform=ccrs.Geodetic())
    ax.plot('lat', 'lon', data=df.where(df.color=="red"), c="red", transform=ccrs.Geodetic())
    
    ax.set_title(f'Traffic across time in Bergen: {str(pd.Timestamp.tz_convert(list(df_dict2.keys())[i], pytz.timezone("Europe/Oslo")))[:16]}')
    print(f"Time for plotting {i}: {time()-ts}")
ani = FuncAnimation(fig=fig, 
                    func=animate, 
                    frames=len(df_dict2.keys()), 
                    interval=1000, 
                    repeat_delay=False,
                    cache_frame_data=False)

ani.save(rf'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\08_reporting\AnimatedPlot{day.replace("_", "")}morning.gif', 
         dpi=200)
