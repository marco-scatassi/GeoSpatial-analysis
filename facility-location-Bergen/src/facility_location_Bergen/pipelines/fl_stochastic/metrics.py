import sys
sys.path.append(r'\\Pund\Stab$\guest801981\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')
# sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')
import warnings
from shapely.errors import ShapelyDeprecationWarning
# Ignore the ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import os
import dill
import random
import numpy as np
import pandas as pd
import pickle as pkl
import geopandas as gpd
from shapely.geometry import Point
from sputility import evaluate_stochastic_solution
from retrieve_global_parameters import (retrieve_adj_matrix_path, 
                                        retrieve_average_graph_path,
                                        retrieve_light_solution_path)
from log import print_INFO_message_timestamp, print_INFO_message
from facility_location import (StochasticFacilityLocationMetrics,
                               StochasticFacilityLocation, 
                               FacilityLocation,
                               AdjacencyMatrix)

print_INFO_message_timestamp(f"START...")

ROOTH = r"\/Pund/Stab$/guest801981/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/"

## Useful functions
def sample_idx(idxs, sample_ratio=0.1):
    n = len(idxs)
    s = random.sample(idxs, int(n*sample_ratio))
    return s
def sample_matrix(matrix, idx_sample):
    n_sample = len(idx_sample)
    sample_matrix = np.zeros((n_sample, n_sample))
    for i, r in enumerate(matrix[idx_sample, :]):
        sample_matrix[i,:] = r[idx_sample]
        
    return sample_matrix
def sample_coords(coordinates, idx_sample):
    sample_coords = coordinates.iloc[idx_sample]
    return sample_coords


##################################### INITIALIZATION #####################################
method = "LS"
max_iter = 25

times = ["morning", "midday", "afternoon"]
times2 = ["morning", "midday", "afternoon", "all_day"]
average_graphs = {}

for time in times2:
    print_INFO_message(f"Loading avg graph for {time}")
    path = ROOTH + retrieve_average_graph_path(time, connected=True, splitted=True, firstSCC=True)
    with open(path, "rb") as f:
        if time == "all_day":
            average_graphs["all_day_free_flow"] = pkl.load(f)
        else:
            average_graphs[time] = pkl.load(f)

adj_paths = {time: ROOTH + retrieve_adj_matrix_path(time) for time in times2}
adj_matricies = {time: None for time in average_graphs.keys()}

print_INFO_message(f"Loading adj matrices")
for time in adj_matricies.keys():
    print_INFO_message(f"Loading adj matrix for {time}")
    if time == "all_day_free_flow":
        with open(ROOTH + retrieve_adj_matrix_path("all_day", free_flow=True), "rb") as f:
            adj_matricies["all_day_free_flow"] = pkl.load(f)
    else:
        with open(adj_paths[time], "rb") as f:
            adj_matricies[time] = pkl.load(f)
    print_INFO_message(f"Adj matrix for all_day_free_flow")
    
    print_INFO_message(f"Adj matrix for {time} loaded with shape {adj_matricies[time].shape}")


## Problem initialization
#It's not possible to solve the problem exactly using all the nodes in the graph. The problem is too big. We can try to solve it using a subset of the nodes.
random.seed(324324)
RATIO1 = 0.1
RATIO2= 0.05
n_locations = [1,2,3]

idx_sampled = sample_idx(list(range(len(average_graphs["morning"].nodes()))), RATIO1)
idx_sampled2 = sample_idx(idx_sampled, RATIO2)
        
coordinates = {time: pd.Series(list(average_graphs[time].nodes())) for time in adj_matricies.keys()}

keys = list(adj_matricies.keys())

for time in keys:
    coordinates[time] = coordinates[time].apply(lambda x: Point(x))
    coordinates[time] = gpd.GeoDataFrame(geometry=coordinates[time])
    coordinates[time]["geometry_x"] = coordinates[time].geometry.x
    coordinates[time]["geometry_y"] = coordinates[time].geometry.y
    coordinates[time].sort_values(by=["geometry_x", "geometry_y"], inplace=True)
    coordinates[time].drop(columns=["geometry_x", "geometry_y"], inplace=True)

coordinates_sampled = {time: sample_coords(coordinates[time], idx_sampled) for time in keys}
coordinates_sampled2 = {time: sample_coords(coordinates[time], idx_sampled2) for time in keys}

coordinates_index_sampled = {time: coordinates_sampled[time].index for time in keys}
coordinates_index_sampled2 = {time: coordinates_sampled2[time].index for time in keys}

adj_matricies_sampled = {time: adj_matricies[time][coordinates_index_sampled2[time], :][:, coordinates_index_sampled[time]] for time in keys}
print(f"adj_matricies_sampled.shape: {adj_matricies_sampled['morning'].shape}")

weighted_adj_matricies = {time: AdjacencyMatrix(adj_matrix=adj_matricies_sampled[time],
                                  kind="empirical",
                                  epsg=None,
                                  mode="time") for time in times}

custom_scenario = AdjacencyMatrix(adj_matrix=adj_matricies_sampled["all_day_free_flow"],
                                    kind="empirical",
                                    epsg=None,
                                    mode="time")

scenarioProbabilities = {time: 1/len(weighted_adj_matricies) for time in times}

##################################### METRICS COMPUTATION #####################################
print_INFO_message_timestamp(f"Computing metrics")
for i, m in enumerate(n_locations):
    print_INFO_message_timestamp(f"Initializing data for {m} locations")

    #---- scenario data ----#
    path = ROOTH + rf"data/07_model_output/{m}_locations/stochastic_solution/lshape_solution.pkl"
    
    RP = StochasticFacilityLocation.load(path)
    RP_metrics = StochasticFacilityLocationMetrics(RP)
    
    print_INFO_message_timestamp(f"Loading deterministic solutions for {m} locations")
    fls_deterministics = {}
    for time in times:
        print_INFO_message(f"Loading deterministic solution for {time}")
        path = ROOTH + retrieve_light_solution_path(m, time)
        fls_deterministics[time] = FacilityLocation.load(path)
        
    print_INFO_message(f"Computing metrics for {m} locations")
    if i == 0:
        df_metrics = RP_metrics.evaluate_stochastic_solution(
            scenarios_data=weighted_adj_matricies,
            scenarioProbabilities=scenarioProbabilities,
            fls_deterministics=fls_deterministics,
            df = pd.DataFrame(columns=['n_locations', 'RP', 'RP_Out_of_Sample', 'WS', 'EVPI', 'VSS']),
            method="LS", 
            max_iter=max_iter,
            custom_scenario=custom_scenario
        )
    else:
        df_metrics = RP_metrics.evaluate_stochastic_solution(
            scenarios_data=weighted_adj_matricies,
            scenarioProbabilities=scenarioProbabilities,
            fls_deterministics=fls_deterministics,
            df = df_metrics,
            method="LS", 
            max_iter=max_iter,
            custom_scenario=custom_scenario
        )
    print_INFO_message_timestamp(f"Metrics for {m} locations computed")
    print_INFO_message(f"Saving metrics for {m} locations")
    df_metrics.to_csv(ROOTH + rf"data/07_model_output/stochastic_solution_evaluation_metrics_{m}.csv", index=False)
    print_INFO_message(f"Metrics for {m} locations saved")
        
print_INFO_message(f"Saving metrics")
try:
    saving_path = ROOTH + rf"/data/07_model_output/stochastic_solution_evaluation_metrics.csv"
    df_metrics.to_csv(saving_path, index=False)
except:
    print_INFO_message(f"Path {saving_path} not valid")
    saving_path = input("Insert the path where you want to save the metrics: ")
    df_metrics.to_csv(saving_path, index=False)