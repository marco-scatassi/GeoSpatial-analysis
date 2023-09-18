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
from log import print_INFO_message_timestamp, print_INFO_message
from facility_location import AdjacencyMatrix, StochasticFacilityLocation
from retrieve_global_parameters import retrieve_adj_matrix_path, retrieve_average_graph_path

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


times = ["morning", "midday", "afternoon"]
average_graphs = {}
time = "all_day"

print_INFO_message(f"Loading avg graph for {time}")
path = ROOTH + retrieve_average_graph_path(time, connected=True, splitted=True, firstSCC=True)
with open(path, "rb") as f:
    average_graphs[time] = pkl.load(f)
adj_paths = {time: ROOTH + retrieve_adj_matrix_path(time) for time in times}
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
ratio1 = 0.1
ratio2= 0.05

idx_sampled = sample_idx(list(range(len(average_graphs["all_day"].nodes()))), ratio1)
idx_sampled2 = sample_idx(idx_sampled, ratio2)

coordinates = pd.Series(list(average_graphs["all_day"].nodes()))
coordinates = coordinates.apply(lambda x: Point(x))
coordinates = gpd.GeoDataFrame(geometry=coordinates)
coordinates["geometry_x"] = coordinates.geometry.x
coordinates["geometry_y"] = coordinates.geometry.y
coordinates.sort_values(by=["geometry_x", "geometry_y"], inplace=True)
coordinates.drop(columns=["geometry_x", "geometry_y"], inplace=True)

coordinates_sampled = sample_coords(coordinates, idx_sampled)
coordinates_sampled2 = sample_coords(coordinates, idx_sampled2)

print_INFO_message_timestamp(f"coordinates_sampled shape: {coordinates_sampled.shape}"+
                             f"\ncoordinates_sampled2 shape: {coordinates_sampled2.shape}")

n_locations = [1,2,3]

for m in n_locations:
    probabilities = {time: 1/len(weighted_adj_matricies) for time in times}
    fl_stochastic = StochasticFacilityLocation(coordinates=coordinates_sampled,
                                            n_of_locations_to_choose=m,
                                            candidate_coordinates=coordinates_sampled2,)
    fl_stochastic.solve(scenarios_data=weighted_adj_matricies,
                        scenarioProbabilities=probabilities,
                        method="EF",
                        max_iter=20,)

    print_INFO_message_timestamp(f"Saving solution")
    fl_stochastic.save(ROOTH + rf"data/07_model_output/{m}_locations/stochastic_solution/lshape_solution.pkl")