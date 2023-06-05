import sys

sys.path.append(
    r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
)

import copy
import pickle
import regex as re
import pandas as pd
import datetime as dt
import geopandas as gpd
from kedro.extras.datasets.pickle import PickleDataSet
from log import print_INFO_message_timestamp, print_INFO_message
from retrieve_global_parameters import (
    retrieve_gdf_average_path,
    retrieve_gdf_path,
    retrieve_catalog_path,
)

################################################################################## STEP 1 ######################################################################

# --------------------------------------- create the average gdf ---------------------------------------
def get_time_gdf(gdf, time="morning"):
    first_morning_time = dt.time(5, 30)
    last_morning_time = dt.time(7, 30)
    first_midday_time = dt.time(9, 0)
    last_midday_time = dt.time(10, 0)
    first_afternoon_time = dt.time(13, 0)
    last_afternoon_time = dt.time(15, 0)

    if time == "morning":
        p1 = gdf["api_call_time"].dt.time >= first_morning_time
        p2 = gdf["api_call_time"].dt.time <= last_morning_time
        return gdf[p1 & p2]
    elif time == "midday":
        p1 = gdf["api_call_time"].dt.time >= first_midday_time
        p2 = gdf["api_call_time"].dt.time <= last_midday_time
        return gdf[p1 & p2]
    elif time == "afternoon":
        p1 = gdf["api_call_time"].dt.time >= first_afternoon_time
        p2 = gdf["api_call_time"].dt.time <= last_afternoon_time
        return gdf[p1 & p2]
    else:
        raise ValueError("time must be 'morning', 'midday' or 'afternoon'")


# --------------------------------------- create the average gdf ---------------------------------------
def create_average_gdf_sub(gdf):
    gdf = gdf[gdf["currentFlow.traversability"] == "open"]

    gdf["geometry_length"] = gdf["geometry_length"].apply(
        lambda x: tuple(x) if type(x) == list else tuple([x])
    )
    gdf_line = gdf.set_geometry("geometry.multi_line")
    gdf_point = gdf.set_geometry("geometry.extreme_point")

    gdf_line_average = (
        gdf_line.groupby(
            ["description", "geometry_length", "geometry.multi_line"],
            sort=False,
            dropna=False,
        )[
            [
                "length",
                "currentFlow.speed",
                "currentFlow.speedUncapped",
                "currentFlow.freeFlow",
                "currentFlow.jamFactor",
                "currentFlow.confidence",
            ]
        ]
        .mean()
        .reset_index()
    )

    gdf_point_average = (
        gdf_point.groupby(
            ["description", "geometry_length", "geometry.extreme_point"],
            sort=False,
            dropna=False,
        )[
            [
                "length",
                "currentFlow.speed",
                "currentFlow.speedUncapped",
                "currentFlow.freeFlow",
                "currentFlow.jamFactor",
                "currentFlow.confidence",
            ]
        ]
        .mean()
        .reset_index()
    )

    gdf_average = copy.deepcopy(gdf_point_average)
    gdf_average["geometry.multi_line"] = gdf_line_average["geometry.multi_line"]

    gdf_average = gdf_average[
        [
            "description",
            "geometry_length",
            "geometry.extreme_point",
            "geometry.multi_line",
            "length",
            "currentFlow.speed",
            "currentFlow.speedUncapped",
            "currentFlow.freeFlow",
            "currentFlow.jamFactor",
            "currentFlow.confidence",
        ]
    ]

    return gdf_average


# --------------------------------------- merge the average gdfs ---------------------------------------
def merge_average_gdf(gdfs_average):
    keys = list(gdfs_average.keys())
    gdf_merge = gdfs_average[keys[0]][
        ["description", "geometry_length", "geometry.extreme_point"]
    ].merge(
        gdfs_average[keys[1]][
            ["description", "geometry_length", "geometry.extreme_point"]
        ],
        how="inner",
        on=["description", "geometry_length", "geometry.extreme_point"],
    )
    for i in range(2, len(keys)):
        try:
            gdf_merge = gdf_merge.merge(
                gdfs_average[keys[i]][
                    ["description", "geometry_length", "geometry.extreme_point"]
                ],
                how="inner",
                on=["description", "geometry_length", "geometry.extreme_point"],
            )
        except:
            print_INFO_message(f"Could not merge {keys[i]}")
            return gdf_merge
    return gdf_merge


# --------------------------------------- create the average gdf ---------------------------------------
def create_average_gdfs(days):
    finished = False
    gdfs = []
    for day in days:
        try:
            gdf_path = retrieve_gdf_path(day, processed=True)
        except:
            raise ValueError(f"Could not retrieve gdf_path for day {day}")

        with open(gdf_path, "rb") as f:
            gdfs.append(pickle.load(f))

    gdf = pd.concat(gdfs, ignore_index=True)
    gdf_morning = get_time_gdf(gdf, time="morning")
    gdf_midday = get_time_gdf(gdf, time="midday")
    gdf_afternoon = get_time_gdf(gdf, time="afternoon")

    gdf_average = create_average_gdf_sub(gdf)
    gdf_morning_average = create_average_gdf_sub(gdf_morning)
    gdf_midday_average = create_average_gdf_sub(gdf_midday)
    gdf_afternoon_average = create_average_gdf_sub(gdf_afternoon)

    gdfs_average = {
        "all_day": gdf_average,
        "morning": gdf_morning_average,
        "midday": gdf_midday_average,
        "afternoon": gdf_afternoon_average,
    }

    gdf_merge = merge_average_gdf(gdfs_average)

    gdfs_average_merged = {}
    for key in gdfs_average.keys():
        gdfs_average_merged[key] = gdfs_average[key].merge(
            gdf_merge,
            how="inner",
            on=["description", "geometry_length", "geometry.extreme_point"],
        )

    for key in gdfs_average_merged.keys():
        # save gdf
        saving_path = retrieve_gdf_average_path(key)
        dataset = PickleDataSet(filepath=saving_path)
        dataset.save(gdfs_average_merged[key])

    finished = True

    return finished


# ---------------------------------------------- update_data_catalog ---------------------------------------
def update_data_catalog_gdf(trigger):
    finished = False
    if trigger:
        catalog_path = retrieve_catalog_path()

        with open(catalog_path, "r+") as f:
            contents = f.read()
            for key in ["all_day", "morning", "midday", "afternoon"]:
                result = re.search(rf"gdf_average_{key}:", contents)
                gdf_path = retrieve_gdf_average_path(key)
                if result is None:
                    contents = "\n".join(
                        [
                            contents,
                            "\n    ".join(
                                [
                                    f"create_average_gdfs.{key}.gdf_average_{key}:",
                                    f"type: pickle.PickleDataSet",
                                    f"filepath: {gdf_path}",
                                ]
                            ),
                        ]
                    )

            f.seek(0)
            f.truncate()
            f.write(contents)

        finished = True

    return finished
