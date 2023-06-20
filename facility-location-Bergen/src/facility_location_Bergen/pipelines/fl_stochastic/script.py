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
import geopandas as gpd
from shapely.geometry import Point
from log import print_INFO_message_timestamp, print_INFO_message
from facility_location import AdjacencyMatrix, StochasticFacilityLocation
from retrieve_global_parameters import retrieve_adj_matrix_path
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
times = ["morning", "midday"]#, "afternoon"]
average_graphs = {}
time = "all_day"

print_INFO_message(f"Loading avg graph for {time}")
path = rf"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\03_primary\average_graph_{time}.pkl"
with open(path, "rb") as f:
    average_graphs[time] = pkl.load(f)
adj_paths = {time: r"C:/Users/Marco/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/" + retrieve_adj_matrix_path(time) for time in times}
adj_matricies = {time: None for time in times}

print_INFO_message(f"Loading adj matrices")
for time in times:
    print_INFO_message(f"Loading adj matrix for {time}")
    with open(adj_paths[time], "rb") as f:
        adj_matricies[time] = pkl.load(f)
    print_INFO_message(f"Adj matrix for {time} loaded with shape {adj_matricies[time].shape}")


weighted_adj_matricies = {time: AdjacencyMatrix(adj_matrix=adj_matricies[time],
                                  kind="empirical",
                                  epsg=None,
                                  mode="time") for time in times}
## Problem initialization
#It's not possible to solve the problem exactly using all the nodes in the graph. The problem is too big. We can try to solve it using a subset of the nodes.
random.seed(324324)
ratio1 = 1/50
ratio2= 1/20
coordinates = pd.Series(list(average_graphs["all_day"].nodes()))
coordinates = coordinates.apply(lambda x: Point(x))
coordinates = gpd.GeoDataFrame(geometry=coordinates)
idx_sampled = sample_idx(list(coordinates.index), ratio1)
idx_sampled2 = sample_idx(idx_sampled, ratio2)
coordinates_sampled = sample_coords(coordinates, idx_sampled)
coordinates_sampled2 = sample_coords(coordinates, idx_sampled2)

print_INFO_message_timestamp(f"coordinates_sampled shape: {coordinates_sampled.shape}"+
                             f"\ncoordinates_sampled2 shape: {coordinates_sampled2.shape}")

n_locations = 1
probabilities = {time: 1/len(weighted_adj_matricies) for time in times}
fl_stochastic = StochasticFacilityLocation(coordinates=coordinates_sampled,
                                           n_of_locations_to_choose=n_locations,
                                           candidate_coordinates=coordinates_sampled2,)
fl_stochastic.solve(scenarios_data=weighted_adj_matricies,
                    scenarioProbabilities=probabilities,
                    method="LS",
                    max_iter=2,)

print_INFO_message_timestamp(f"Saving solution")
fl_stochastic.save(rf"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\07_model_output\{n_locations}_locations\stochastic_solution\extensive_form_solution.pkl")