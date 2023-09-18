import sys
sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')
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
from sputility import evaluate_stochastic_solution
from retrieve_global_parameters import retrieve_adj_matrix_path, retrieve_average_graph_path
from log import print_INFO_message_timestamp, print_INFO_message
from facility_location import (StochasticFacilityLocationMetrics,
                               StochasticFacilityLocation, 
                               FacilityLocation,
                               AdjacencyMatrix)

print_INFO_message_timestamp(f"START...")

ROOTH = r"\/Pund/Stab$/guest801981/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/"

##################################### INITIALIZATION #####################################
method = "LS"
max_iter = 25
time = "all_day"
average_graphs = {}
n_locations = [1,2,3]
scenario_names = ["morning", "midday", "afternoon"]

print_INFO_message(f"Loading avg graph for {time}")
path = ROOTH + retrieve_average_graph_path(time, connected=True, splitted=True, firstSCC=True)
with open(path, "rb") as f:
    average_graphs[time] = pkl.load(f)
adj_paths = {time: ROOTH + retrieve_adj_matrix_path(time) for time in scenario_names}
adj_matricies = {time: None for time in scenario_names}

print_INFO_message(f"Loading adj matrices")
for time in scenario_names:
    print_INFO_message(f"Loading adj matrix for {time}")
    with open(adj_paths[time], "rb") as f:
        adj_matricies[time] = pkl.load(f)
    print_INFO_message(f"Adj matrix for {time} loaded with shape {adj_matricies[time].shape}")


weighted_adj_matricies = {time: AdjacencyMatrix(adj_matrix=adj_matricies[time],
                                  kind="empirical",
                                  epsg=None,
                                  mode="time") for time in scenario_names}

scenarioProbabilities = {time: 1/len(weighted_adj_matricies) for time in scenario_names}

##################################### METRICS COMPUTATION #####################################
print_INFO_message_timestamp(f"Computing metrics")
for i, m in enumerate(n_locations):
    print_INFO_message_timestamp(f"Initializing data for {m} locations")

    #---- scenario data ----#
    root = rf"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\07_model_output"
    path = root + rf"\{m}_locations\stochastic_solution\lshape_solution.pkl"
    
    RP = StochasticFacilityLocation.load(path)
    RP_metrics = StochasticFacilityLocationMetrics(RP)
    
    print_INFO_message_timestamp(f"Loading deterministic solutions for {m} locations")
    fls_deterministics = {}
    for time in scenario_names:
        print_INFO_message(f"Loading deterministic solution for {time}")
        path = root + rf"\{m}_locations\deterministic_exact_solutions\light_exact_solution_{time}.pkl"
        fls_deterministics[time] = FacilityLocation.load(path)
        
    print_INFO_message(f"Computing metrics for {m} locations")
    if i == 0:
        df_metrics = RP_metrics.evaluate_stochastic_solution(
            scenarios_data=weighted_adj_matricies,
            scenarioProbabilities=scenarioProbabilities,
            fls_deterministics=fls_deterministics,
            df = pd.DataFrame(columns=['n_locations', 'RP', 'RP_Out_of_Sample', 'WS', 'EVPI', 'VSS']),
            method="LS", 
            max_iter=max_iter
        )
    else:
        df_metrics = RP_metrics.evaluate_stochastic_solution(
            scenarios_data=weighted_adj_matricies,
            scenarioProbabilities=scenarioProbabilities,
            fls_deterministics=fls_deterministics,
            df = df_metrics,
            method="LS", 
            max_iter=max_iter
        )
    print_INFO_message_timestamp(f"Metrics for {m} locations computed")
    print_INFO_message(f"Saving metrics for {m} locations")
    df_metrics.to_csv(root + rf"\stochastic_solution_evaluation_metrics_{m}.csv", index=False)
    print_INFO_message(f"Metrics for {m} locations saved")
        
print_INFO_message(f"Saving metrics")
try:
    saving_path = root + rf"\stochastic_solution_evaluation_metrics.csv"
    df_metrics.to_csv(saving_path, index=False)
except:
    print_INFO_message(f"Path {saving_path} not valid")
    saving_path = input("Insert the path where you want to save the metrics: ")
    df_metrics.to_csv(saving_path, index=False)