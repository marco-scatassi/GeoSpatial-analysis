## Import
import sys

sys.path.append(
    r"\\Pund\Stab$\guest801981\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
)

import warnings
from shapely.errors import ShapelyDeprecationWarning

# Ignore the ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
import numpy as np
import regex as re
import pandas as pd
import pickle as pkl
import networkx as nx
from shapely.geometry import Point
from facility_location import FacilityLocation
from kedro.extras.datasets.pickle import PickleDataSet
from graph_manipulation import create_gdf_from_mapping
from log import print_INFO_message_timestamp, print_INFO_message
from retrieve_global_parameters import *

ROOT = r"\/Pund/Stab$/guest801981/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/"

################################################################################## STEP 1 ######################################################################

## -------------------------------------------------- verify dataframe already created ------------------------------------------------- ##
def verify_df_already_created(data: dict):
    is_created = False
    # verify if the file already exists
    weight = data["weight"]
    time_solution = data["time_solution"]
    time_scenario = data["time_scenario"]
    facilities_number = data["facilities_number"]
    worst = data["worst"] 
    handpicked = data["handpicked"] 
    fl_class = data["fl_class"]
    saving_path = retrieve_solution_vs_scenario_path(
        facilities_number, time_solution, time_scenario, 
        weight, worst, handpicked, fl_class
    )
    if os.path.exists(saving_path) or os.path.exists(ROOT+saving_path):
        is_created = True

    return is_created


################################################################################## STEP 2 ######################################################################

## ------------------------------------------------------- solution_vs_scenario ------------------------------------------------------- ##
# Function to get the minimum distance between two points avoiding taking 
# into account the carriageway, i.e. the side of the road
def get_min_distance(starting_index, ending_index, geodf, mapping, average_graph, weight):
    distances = {}
    for i in geodf.iloc[starting_index].contained:
        for j in geodf.iloc[ending_index].contained:
            distances[(j, i)] = nx.dijkstra_path_length(
                                G=average_graph, source=mapping[i], target=mapping[j], weight=weight
                            ),
            
    mapped_key = min(distances, key=distances.get)
    min_distance = distances[mapped_key]
    return min_distance, mapped_key


# This function takes as input the time of the solution and the time of the scenario and returns a dataframe with the distance from the
# exact solution to all the other nodes in the graph, under the specified scenario.
def solution_vs_scenario(data, is_created=False):
    weight = data["weight"]
    time_solution = data["time_solution"]
    time_scenario = data["time_scenario"]
    facilities_number = data["facilities_number"]
    worst = data["worst"] 
    handpicked = data["handpicked"]
    fl_class = data["fl_class"]
    is_free_flow = True if time_solution == "all_day_free_flow" else False

    saving_path = retrieve_solution_vs_scenario_path(
        facilities_number, time_solution, time_scenario,
        weight, worst, handpicked, fl_class
    )
    if not os.path.exists(saving_path):
        saving_path = ROOT+saving_path
        
    df_ = PickleDataSet(filepath=saving_path)

    if not is_created:
        # Load the exact solution
        print_INFO_message_timestamp(f"Loading exact solution for {time_solution}")
        path = retrieve_light_solution_path(facilities_number, time_solution, handpicked, fl_class)
        adj_mapping_path = retrieve_adj_mapping_path(time_solution, is_free_flow, handpicked)
        adj_mapping_path_2 = retrieve_adj_mapping_path_2(time_solution, is_free_flow, handpicked)
        try:
            fls_exact_solution = FacilityLocation.load(path)
            with open(adj_mapping_path, "rb") as f:
                adj_mapping = pkl.load(f)
            adj_mapping_reverse = {v: k for k, v in adj_mapping.items()}
            with open(adj_mapping_path_2, "rb") as f:
                adj_mapping_2 = pkl.load(f)
        except:
            fls_exact_solution = FacilityLocation.load(ROOT+path)
            with open(ROOT+adj_mapping_path, "rb") as f:
                adj_mapping = pkl.load(f)
            adj_mapping_reverse = {v: k for k, v in adj_mapping.items()}
            with open(ROOT+adj_mapping_path_2, "rb") as f:
                adj_mapping_2 = pkl.load(f)

        # Load the average graph
        print_INFO_message(f"Loading adj matrix for {time_scenario}")
        if worst:
            path = rf"\\Pund\Stab$\guest801981\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\03_primary\worst_average_graph_{time_scenario}.pkl"
        else:
            path = ROOT+retrieve_average_graph_path(time_scenario, connected=True, splitted=True, firstSCC=True)
        with open(path, "rb") as f:
            average_graph = pkl.load(f)

        # extract the coordinates of the exact solution
        ff_solutions_location = fls_exact_solution.locations_coordinates
    
        # compute the distance from the exact solution to all the other nodes in the graph
        print_INFO_message_timestamp(
            f"Compute the distance from the {time_solution} solution to all the other nodes in the {time_scenario} graph"
        )
        
        # get the real source nodes using the stored mapping
        adj_mapping_point = {}
        for key, value in adj_mapping.items():
            adj_mapping_point[key] = Point(value)
        
        geodf = create_gdf_from_mapping(adj_mapping_point)
        

        temporal_distances = {
            ff_solutions_location[i].geometry.coords[0]: []
            for i in range(len(ff_solutions_location))
        }

        coordinates = [(x,y) for x,y in zip(fls_exact_solution.coordinates.geometry.x, fls_exact_solution.coordinates.geometry.y)]
        
        for i, node in enumerate(average_graph):
            if i % 500 == 0:
                print_INFO_message(f"{i} out of {len(average_graph.nodes)}")

            if node in coordinates:
                keys = list(temporal_distances.keys())
                
                for k in keys:
                    min_dis, mapped_index = get_min_distance(adj_mapping_reverse[k], 
                                             adj_mapping_reverse[node], 
                                             geodf, 
                                             adj_mapping, 
                                             average_graph, 
                                             weight)
                    temporal_distances[k].append(
                        (
                            adj_mapping[mapped_index[1]],
                            node,
                            adj_mapping[mapped_index[0]],
                            min_dis[0],
                        )
                    )
                else:
                    continue

        # create a dataframe with the distance from the exact solution to all the other nodes in the graph
        d = {"source": [], "new_source": [], "target": [], "new_target": [], "travel_time": []}

        for key, value in temporal_distances.items():
            for new_source, target, new_target, distance  in value:
                d["source"].append(key)
                d["new_source"].append(new_source)
                d["target"].append(target)
                d["new_target"].append(new_target)
                d["travel_time"].append(round(distance / 60, 3))
        
        print(pd.DataFrame(d))
        df_.save(pd.DataFrame(d))

    return df_


################################################################################## STEP 3 ######################################################################

# ---------------------------------------------- update_data_catalog ---------------------------------------
def update_data_catalog_gdf(df_pkl):
    finished = False
    if df_pkl is not None:
        catalog_path = retrieve_catalog_path()
        solution_vs_scenario_path = df_pkl._filepath
        solution_vs_scenario_name = str(solution_vs_scenario_path).split("/")[-1][:-4]

        with open(catalog_path, "r+") as f:
            contents = f.read()
            result = re.search(rf"{solution_vs_scenario_name}:", contents)
            if result is None:
                contents = "\n".join(
                    [
                        contents,
                        "\n    ".join(
                            [
                                f"solution_comparison.{solution_vs_scenario_name}:",
                                f"type: pickle.PickleDataSet",
                                f"filepath: {solution_vs_scenario_path}",
                            ]
                        ),
                    ]
                )

            f.seek(0)
            f.truncate()
            f.write(contents)

        finished = True

    return finished
