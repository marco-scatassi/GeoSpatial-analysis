import sys

sys.path.append(
    r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
)

import warnings
from shapely.errors import ShapelyDeprecationWarning

# Ignore the ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
import copy
import random
import numpy as np
import pandas as pd
import pickle as pkl
import geopandas as gpd
from shapely.geometry import Point
from log import print_INFO_message_timestamp, print_INFO_message
from facility_location import AdjacencyMatrix, FacilityLocation
from retrieve_global_parameters import *

ROOTH = rf"\/Pund/Stab$/guest801981/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/"

## ------------------------------------------------------------- UTILS FUNCTIONS ------------------------------------------------------------- ##


def sample_idx(idxs, sample_ratio=0.1):
    n = len(idxs)
    s = random.sample(idxs, int(n * sample_ratio))
    return s


def sample_matrix(matrix, idx_sample):
    n_sample = len(idx_sample)
    sample_matrix = np.zeros((n_sample, n_sample))
    for i, r in enumerate(matrix[idx_sample, :]):
        sample_matrix[i, :] = r[idx_sample]
    return sample_matrix


def sample_coords(coordinates, idx_sample):
    sample_coords = coordinates.iloc[idx_sample]
    return sample_coords

def sort_coordinates(time, coordinates):
    coordinates[time]["geometry_x"] = coordinates[time].geometry.x
    coordinates[time]["geometry_y"] = coordinates[time].geometry.y
    coordinates[time].sort_values(by=["geometry_x", "geometry_y"], inplace=True)
    coordinates[time].drop(columns=["geometry_x", "geometry_y"], inplace=True)


## ------------------------------------------------------------- PREPARATION ------------------------------------------------------------- ##
def verify_problem_already_solved(fl_data):
    print_INFO_message_timestamp("CHECKING IF PROBLEM HAS BEEN ALREADY SOLVED")
    n_facilities = fl_data["facilities_number"]
    handpicked = fl_data["handpicked"]
    fl_class = fl_data["fl_class"]
    if handpicked:
        path = ROOTH + rf"data/07_model_output/random_candidate_plus_handpicked/{fl_class}/{n_facilities}_locations/deterministic_exact_solutions"
    else:
        path = ROOTH + rf"data/07_model_output/only_random_candidate_location/{fl_class}/{n_facilities}_locations/deterministic_exact_solutions"
    if len(os.listdir(path)) <15:
        return False
    else:
        return True
    
def set_up_fl_problems(fl_data, already_solved):
    if not already_solved:
        print_INFO_message_timestamp("STARTING SCRIPT")
        times = ["all_day", "morning", "midday", "afternoon"]
        average_graphs = {}

        print_INFO_message("Loading average graphs")
        for time in times:
            if time != "all_day_free_flow":
                print_INFO_message(f"Loading adj matrix for {time}")
                path = ROOTH + retrieve_average_graph_path(time, connected=True, splitted=True, firstSCC=True)
                with open(path, "rb") as f:
                    average_graphs[time] = pkl.load(f)

        ADJ_PATHS = {
            time: r"\/Pund/Stab$/guest801981/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/"
            + retrieve_adj_matrix_path(time, free_flow=False, handpicked=fl_data["handpicked"])
            for time in times
        }
        ADJ_PATHS["all_day_free_flow"] = (
            r"\/Pund/Stab$/guest801981/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/"
            + retrieve_adj_matrix_path("all_day", free_flow=True, handpicked=fl_data["handpicked"])
        )
        adj_matricies = {time: None for time in times}

        print_INFO_message("Loading adj matricies")
        for time in times:
            with open(ADJ_PATHS[time], "rb") as f:
                adj_matricies[time] = pkl.load(f)

        with open(ADJ_PATHS["all_day_free_flow"], "rb") as f:
            adj_matricies["all_day_free_flow"] = pkl.load(f)

        weighted_adj_matricies = {
            time: AdjacencyMatrix(
                adj_matrix=adj_matricies[time], kind="empirical", epsg=None, mode="time"
            )
            for time in times
        }

        weighted_adj_matricies["all_day_free_flow"] = AdjacencyMatrix(
            adj_matrix=adj_matricies["all_day_free_flow"],
            kind="empirical",
            epsg=None,
            mode="time",
        )

        ## It's not possible to solve the problem exactly using all the nodes in the graph. The problem is too big. We can try to solve it using a subset of the nodes.
        random.seed(fl_data["seed"])

        RATIO1 = fl_data["ratio1"]
        RATIO2 = fl_data["ratio2"]
        N_LOC_TO_CHOOSE = fl_data["facilities_number"]

        idx_sampled = sample_idx(list(range(len(average_graphs["all_day"].nodes()))), RATIO1)
        idx_sampled2 = sample_idx(idx_sampled, RATIO2)

        original_order_coordinates = {}
        coordinates = {time: pd.Series(list(average_graphs[time].nodes())) for time in times}
        
        for time in times:
            coordinates[time] = coordinates[time].apply(lambda x: Point(x))
            coordinates[time] = gpd.GeoDataFrame(geometry=coordinates[time])
            original_order_coordinates[time] = copy.deepcopy(coordinates[time])
            sort_coordinates(time, coordinates)
            
        coordinates_sampled = {time: sample_coords(coordinates[time], idx_sampled) for time in times}
        coordinates_sampled2 = {time: sample_coords(coordinates[time], idx_sampled2) for time in times}
        
        coordinates_sampled["all_day_free_flow"] = coordinates_sampled["all_day"]
        coordinates_sampled2["all_day_free_flow"] = coordinates_sampled2["all_day"]
        
        original_order_coordinates["all_day_free_flow"] = original_order_coordinates["all_day"]

        if fl_data["handpicked"]:
            extra_locations = []
            for i, node in enumerate(average_graphs["all_day"].nodes()):
                if i in fl_data["handpicked_locations"]:
                    extra_locations.append(Point(node))
                    
            extra_locations_index = {}
            for time in coordinates_sampled.keys():
                extra_locations_index[time] = []
                for p in extra_locations:
                    for i, e in zip(original_order_coordinates[time].index, original_order_coordinates[time].geometry):
                        if e == p:
                            extra_locations_index[time].append(i)
                            
            for time in coordinates_sampled.keys():
                coordinates_sampled[time] = pd.concat([coordinates_sampled[time], 
                                         gpd.GeoDataFrame(geometry=extra_locations, index=extra_locations_index[time])]).\
                                            drop_duplicates(subset=["geometry"])
                coordinates_sampled2[time] = pd.concat([coordinates_sampled2[time],
                                         gpd.GeoDataFrame(geometry=extra_locations, index=extra_locations_index[time])]).\
                                            drop_duplicates(subset=["geometry"])
                
                sort_coordinates(time, coordinates_sampled)
                sort_coordinates(time, coordinates_sampled2)
        
        # adj_sampled = {key: sample_matrix(adj_matricies[key], idx_sampled) for key in adj_matricies.keys()}

        # weighted_adj_sampled = {key: AdjacencyMatrix(adj_matrix=adj_sampled[key],
        #                                   kind="empirical",
        #                                   epsg=None,
        #                                   mode="time") for key in ADJ_PATHS.keys()}

        # fls_gon = {key: FacilityLocation(coordinates_sampled, N_LOC_TO_CHOOSE, weighted_adj_sampled[key]) for key in weighted_adj_sampled.keys()}
        # fls_gon_plus = {key: FacilityLocation(coordinates_sampled, N_LOC_TO_CHOOSE, weighted_adj_sampled[key]) for key in weighted_adj_sampled.keys()}
        fls_exact = {
            key: FacilityLocation(
                coordinates_sampled[key],
                N_LOC_TO_CHOOSE,
                weighted_adj_matricies[key],
                coordinates_sampled2[key],
            )
            for key in weighted_adj_matricies.keys()
        }

    else:
        fls_exact = {}
        
    return fls_exact

## ------------------------------------------------------------- SOLVE ------------------------------------------------------------- ##
def solve_fl_problems(fls_exact, fl_data):
    # for fl_gon in fls_gon.values():
    #     fl_gon.solve(mode = "approx")
    # print_INFO_message_timestamp("Objective value for the GON approximation")

    # for time, fl_gon in fls_gon.items():
    #     print_INFO_message(f"{time}: {round(fl_gon.solution_value/60, 3)} minutes")

    # for fl_gon_plus in fls_gon_plus.values():
    #     fl_gon_plus.solve(mode = "approx",
    #                       algorithm = "gon_plus",
    #                       n_trial = len(coordinates_sampled))
    # print_INFO_message_timestamp("Objective value for the GON approximation")

    # for time, fl_gon_plus in fls_gon_plus.items():
    #     print_INFO_message(f"{time}: {round(fl_gon_plus.solution_value/60, 3)} minutes")

    solve_list = ["all_day_free_flow", "all_day", "morning", "midday", "afternoon"]
    # solve_list = ["morning"]#, "midday", "afternoon"]
    
    if fls_exact != {}:
        for i, (time, fl_exact) in enumerate(zip(list(fls_exact.keys()), list(fls_exact.values()))):
            if fl_data["handpicked"]:
                root_path = ROOTH + rf"data/07_model_output/random_candidate_plus_handpicked/{fl_data['fl_class']}/{fl_data['facilities_number']}_locations/deterministic_exact_solutions"
            else:
                root_path = ROOTH + rf"data/07_model_output/only_random_candidate_location/{fl_data['fl_class']}/{fl_data['facilities_number']}_locations/deterministic_exact_solutions"
            
            saving_path = root_path + rf"/exact_solution_{time}.pkl"
            saving_path_light = root_path + rf"/light_exact_solution_{time}.pkl"
            saving_path_super_light = root_path + rf"/super_light_exact_solution_{time}.pkl"
            if os.path.exists(saving_path):
                print_INFO_message(f"Exact solution for {time} already exists. Skipping...")
            else:
                if time in solve_list:
                    print_INFO_message_timestamp(f"Solving exact solution for {time}")
                    fl_exact.solve(mode="exact", fl_class=fl_data["fl_class"])
                    print_INFO_message(f"Saving exact solution for {time}")
                    fl_exact.save(saving_path)
                    print_INFO_message(f"Exact solution for {time} SAVED")
                    fl_exact_light = copy.deepcopy(fl_exact)
                    del fl_exact_light.adjacency_matrix
                    del fl_exact_light.model
                    del fl_exact_light.instance
                    fl_exact_light.save(saving_path_light)
                    print_INFO_message(f"Light exact solution for {time} SAVED")
                    fl_exact_super_light = copy.deepcopy(fl_exact_light)
                    del fl_exact_super_light.result
                    fl_exact_super_light.save(saving_path_super_light)
                    print_INFO_message(f"Super light exact solution for {time} SAVED")
                else:
                    print_INFO_message(f"Skipping {time}")
            
    return "Done"
