## Import
import sys

sys.path.append(
    r"\\Pund\Stab$\guest801968\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
)

import warnings
from shapely.errors import ShapelyDeprecationWarning

# Ignore the ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
import regex as re
import pandas as pd
import pickle as pkl
import networkx as nx
from facility_location import FacilityLocation
from kedro.extras.datasets.pickle import PickleDataSet
from log import print_INFO_message_timestamp, print_INFO_message
from retrieve_global_parameters import (
    retrieve_catalog_path,
    retrieve_solution_path,
    retrieve_solution_vs_scenario_path,
)

################################################################################## STEP 1 ######################################################################

## -------------------------------------------------- verify dataframe already created ------------------------------------------------- ##
def verify_df_already_created(data: dict):
    is_created = False
    # verify if the file already exists
    weight = data["weight"]
    time_solution = data["time_solution"]
    time_scenario = data["time_scenario"]
    facilities_number = data["facilities_number"]
    worst = True if data["worst"] == "True" else False
    saving_path = retrieve_solution_vs_scenario_path(
        facilities_number, time_solution, time_scenario, weight, worst
    )
    if os.path.exists(saving_path):
        is_created = True

    return is_created


################################################################################## STEP 2 ######################################################################

## ------------------------------------------------------- solution_vs_scenario ------------------------------------------------------- ##
# This function takes as input the time of the solution and the time of the scenario and returns a dataframe with the distance from the
# exact solution to all the other nodes in the graph, under the specified scenario.


def solution_vs_scenario(data, is_created=False):
    weight = data["weight"]
    time_solution = data["time_solution"]
    time_scenario = data["time_scenario"]
    facilities_number = data["facilities_number"]
    worst = True if data["worst"] == "True" else False

    saving_path = retrieve_solution_vs_scenario_path(
        facilities_number, time_solution, time_scenario, weight, worst
    )
    df_ = PickleDataSet(filepath=saving_path)

    if not is_created:
        # Load the exact solution
        print_INFO_message_timestamp(f"Loading exact solution for {time_solution}")
        path = retrieve_solution_path(facilities_number, time_solution)
        fls_exact_solution = FacilityLocation.load(path)

        # Load the average graph
        print_INFO_message(f"Loading adj matrix for {time_scenario}")
        if worst:
            path = rf"\\Pund\Stab$\guest801968\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\03_primary\worst_average_graph_{time_scenario}.pkl"
        else:
            path = rf"\\Pund\Stab$\guest801968\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\03_primary\average_graph_{time_scenario}_connected_splitted_firstSCC.pkl"
        with open(path, "rb") as f:
            average_graph = pkl.load(f)

        # extract the coordinates of the exact solution
        ff_solutions_location = fls_exact_solution.locations_coordinates

        # compute the distance from the exact solution to all the other nodes in the graph
        print_INFO_message_timestamp(
            f"Compute the distance from the {time_solution} solution to all the other nodes in the {time_scenario} graph"
        )

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
                    temporal_distances[k].append(
                        (
                            node,
                            nx.dijkstra_path_length(
                                G=average_graph, source=k, target=node, weight=weight
                            ),
                        )
                    )
                else:
                    continue

        # create a dataframe with the distance from the exact solution to all the other nodes in the graph
        d = {"source": [], "target": [], "travel_time": []}

        for key, value in temporal_distances.items():
            for node, distance in value:
                d["source"].append(key)
                d["target"].append(node)
                d["travel_time"].append(round(distance / 60, 3))

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
