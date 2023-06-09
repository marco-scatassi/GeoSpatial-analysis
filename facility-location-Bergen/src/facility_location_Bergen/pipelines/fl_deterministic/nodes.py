import sys

sys.path.append(
    r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
)

import warnings
from shapely.errors import ShapelyDeprecationWarning

# Ignore the ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
import random
import numpy as np
import pandas as pd
import pickle as pkl
import geopandas as gpd
from shapely.geometry import Point
from log import print_INFO_message_timestamp, print_INFO_message
from facility_location import AdjacencyMatrix, FacilityLocation
from retrieve_global_parameters import retrieve_adj_matrix_path


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


## ------------------------------------------------------------- PREPARATION ------------------------------------------------------------- ##
def verify_problem_already_solved(fl_data):
    print_INFO_message_timestamp("CHECKING IF PROBLEM HAS BEEN ALREADY SOLVED")
    n_facilities = fl_data["facilities_number"]
    path = rf"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\07_model_output\{n_facilities}_locations\deterministic_exact_solutions"
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
                path = rf"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\03_primary\average_graph_{time}.pkl"
                with open(path, "rb") as f:
                    average_graphs[time] = pkl.load(f)

        ADJ_PATHS = {
            time: r"C:/Users/Marco/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/"
            + retrieve_adj_matrix_path(time)
            for time in times
        }
        ADJ_PATHS["all_day_free_flow"] = (
            r"C:/Users/Marco/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/"
            + retrieve_adj_matrix_path("all_day", free_flow=True)
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
        
        coordinates = {time: pd.Series(list(average_graphs[time].nodes())) for time in times}
        
        for time in times:
            coordinates[time] = coordinates[time].apply(lambda x: Point(x))
            coordinates[time] = gpd.GeoDataFrame(geometry=coordinates[time])
            coordinates[time]["geometry_x"] = coordinates[time].geometry.x
            coordinates[time]["geometry_y"] = coordinates[time].geometry.y
            coordinates[time].sort_values(by=["geometry_x", "geometry_y"], inplace=True)
            coordinates[time].drop(columns=["geometry_x", "geometry_y"], inplace=True)
            
        coordinates_sampled = {time: sample_coords(coordinates[time], idx_sampled) for time in times}
        coordinates_sampled2 = {time: sample_coords(coordinates[time], idx_sampled2) for time in times}
        
        coordinates_sampled["all_day_free_flow"] = coordinates_sampled["all_day"]
        coordinates_sampled2["all_day_free_flow"] = coordinates_sampled2["all_day"]

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
    
    if fls_exact != {}:
        for i, (time, fl_exact) in enumerate(zip(list(fls_exact.keys()), list(fls_exact.values()))):
            saving_path = rf"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\07_model_output\{fl_data['facilities_number']}_locations\deterministic_exact_solutions\exact_solution_{time}.pkl"
            if os.path.exists(saving_path):
                print_INFO_message(f"Exact solution for {time} already exists. Skipping...")
            else:
                if time in solve_list:
                    print_INFO_message_timestamp(f"Solving exact solution for {time}")
                    fl_exact.solve(mode="exact")
                    print_INFO_message(f"{time}")
                    fl_exact.save(saving_path)
                else:
                    print_INFO_message(f"Skipping {time}")
            
    return "Done"
