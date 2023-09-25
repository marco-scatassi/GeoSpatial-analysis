import sys
sys.path.append(r'\\Pund\Stab$\guest801981\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')
# sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')
import warnings
from shapely.errors import ShapelyDeprecationWarning
# Ignore the ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import os
import dill
import copy
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

def sort_coordinates(time, coordinates):
    coordinates[time]["geometry_x"] = coordinates[time].geometry.x
    coordinates[time]["geometry_y"] = coordinates[time].geometry.y
    coordinates[time].sort_values(by=["geometry_x", "geometry_y"], inplace=True)
    coordinates[time].drop(columns=["geometry_x", "geometry_y"], inplace=True)


def load_data(times, handpicked=True):
    average_graphs = {}

    for time in times:
        print_INFO_message(f"Loading avg graph for {time}")
        path = ROOTH + retrieve_average_graph_path(time, connected=True, splitted=True, firstSCC=True)
        with open(path, "rb") as f:
            average_graphs[time] = pkl.load(f)

    adj_paths = {time: ROOTH + retrieve_adj_matrix_path(time, handpicked=handpicked) for time in times}
    adj_matricies = {time: None for time in times}

    print_INFO_message(f"Loading adj matrices")
    for time in times:
        print_INFO_message(f"Loading adj matrix for {time}")
        with open(adj_paths[time], "rb") as f:
            adj_matricies[time] = pkl.load(f)
        print_INFO_message(f"Adj matrix for {time} loaded with shape {adj_matricies[time].shape}")
      
    return average_graphs, adj_matricies



def main():
    ## Problem initialization
    #It's not possible to solve the problem exactly using all the nodes in the graph. The problem is too big. We can try to solve it using a subset of the nodes.
    random.seed(324324)
    RATIO1 = 0.1
    RATIO2= 0.05
    n_locations = [1,2,3]
    times = ["all_day", "morning", "midday", "afternoon"]
    handpicked = True
    handpicked_locations = [5229, 1688, 1842, 4647, 159, 2075, 2428, 361, 3477, 3745, 4731]
    fl_class = sys.argv[1] if len(sys.argv) > 1 else "p-center"
        
    if fl_class not in ["p-center", "p-median"]:
        raise Exception(f"Argument {fl_class} not valid")
        
    average_graphs, adj_matricies = load_data(times, handpicked)

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

    coordinates_index_sampled = {time: coordinates_sampled[time].index for time in times}
    coordinates_index_sampled2 = {time: coordinates_sampled2[time].index for time in times}
    
    if handpicked:
        extra_locations = []
        for i, node in enumerate(average_graphs["all_day"].nodes()):
            if i in handpicked_locations:
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

    print(coordinates_sampled2["all_day"].index)

    adj_matricies_sampled = {time: adj_matricies[time][coordinates_sampled2[time].index, :][:, coordinates_sampled[time].index] for time in times[1:]}
    print(f"adj_matricies_sampled.shape: {adj_matricies_sampled['morning'].shape}")

    weighted_adj_matricies = {time: AdjacencyMatrix(adj_matrix=adj_matricies_sampled[time],
                                      kind="empirical",
                                      epsg=None,
                                      mode="time") for time in times[1:]}

    print_INFO_message_timestamp(f"coordinates_sampled shape: {coordinates_sampled['all_day'].shape}"+
                                 f"\ncoordinates_sampled2 shape: {coordinates_sampled2['all_day'].shape}")


    for m in n_locations:
        probabilities = {time: 1/len(weighted_adj_matricies) for time in times[1:]}
        fl_stochastic = StochasticFacilityLocation(coordinates=coordinates_sampled["all_day"],
                                                n_of_locations_to_choose=m,
                                                candidate_coordinates=coordinates_sampled2["all_day"],)
        fl_stochastic.solve(scenarios_data=weighted_adj_matricies,
                            scenarioProbabilities=probabilities,
                            method="LS",
                            fl_class=fl_class,
                            max_iter=20,)

        print_INFO_message_timestamp(f"Saving solution")
        if handpicked:
            fl_stochastic.save(ROOTH + rf"data/07_model_output/random_candidate_plus_handpicked/{fl_class}/{m}_locations/stochastic_solution/lshape_solution.pkl")
        else:
            fl_stochastic.save(ROOTH + rf"data/07_model_output/only_random_candidate_location/{fl_class}/{m}_locations/stochastic_solution/lshape_solution.pkl")

if __name__ == "__main__":
    main()