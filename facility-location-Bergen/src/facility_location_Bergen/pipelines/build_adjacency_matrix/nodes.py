import sys

sys.path.append(
    r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
)

import warnings
from shapely.errors import ShapelyDeprecationWarning

# Ignore the ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# Filter out the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

import copy
import random
import pickle
import numpy as np
import regex as re
import pandas as pd
import pickle as pkl
import datetime as dt
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from kedro.extras.datasets.pickle import PickleDataSet
from convert_geometry import toMultiLineString, toExtremePoints
from log import print_INFO_message_timestamp, print_INFO_message
from graph_manipulation import create_gdf_from_mapping
from retrieve_global_parameters import *

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

# Function to get the minimum distance between two points avoiding taking 
# into account the carriageway, i.e. the side of the road

def get_min_distance(starting_index, ending_index, geodf, sp, mapping):
    distances = {}
    for i in geodf.iloc[starting_index].contained:
        for j in geodf.iloc[ending_index].contained:
            distances[(i, j)] = sp[mapping[i]][mapping[j]]
            
    mapped_key = min(distances, key=distances.get)
    min_distance = distances[mapped_key]
    return min_distance, mapped_key

##################################################################### STEP 1 #####################################################################

def build_adj_matrix(time):
    finished = False
    average_graph_path = retrieve_average_graph_path(time, True, True, True, False)

    with open(average_graph_path, "rb") as f:
        average_graph = pickle.load(f)
        
    mapping = {}
    mapping_points = {}
    for i, node in enumerate(average_graph.nodes()):
        mapping_points[i] = Point(node)
        mapping[i] = node
        
    geodf = create_gdf_from_mapping(mapping_points)
    
    random.seed(324324)

    RATIO1 = 0.1
    RATIO2 = 0.05

    idx_sampled = sample_idx(list(range(len(average_graph.nodes()))), RATIO1)
    idx_sampled2 = sample_idx(idx_sampled, RATIO2)
        
    coordinates = pd.Series(list(average_graph.nodes()))
    coordinates = coordinates.apply(lambda x: Point(x))
    coordinates = gpd.GeoDataFrame(geometry=coordinates)
    coordinates["geometry_x"] = coordinates.geometry.x
    coordinates["geometry_y"] = coordinates.geometry.y
    coordinates.sort_values(by=["geometry_x", "geometry_y"], inplace=True)
    coordinates.drop(columns=["geometry_x", "geometry_y"], inplace=True)
            
    coordinates_sampled = sample_coords(coordinates, idx_sampled)
    coordinates_sampled2 = sample_coords(coordinates, idx_sampled2)

    # if time == "all_day":
    #     sp_free_flow = dict(
    #         nx.all_pairs_dijkstra_path_length(average_graph, weight="weight2")
    #     )
    #     adj_matrix = np.zeros((len(sp_free_flow), len(sp_free_flow)))
    #     adj_matrix_mapping = {}

    #     print_INFO_message_timestamp("Creating free_flow distance matrix")
    #     for i in range(len((sp_free_flow))):
    #         if mapping_points[i] in coordinates_sampled2.geometry:
    #             for j in range(len(sp_free_flow)):
    #                 if mapping_points[j] in coordinates_sampled.geometry:
    #                     min_dis, mapped_key = get_min_distance(i, j, geodf, sp_free_flow, mapping)
    #                     adj_matrix[i,j] = min_dis
    #                     adj_matrix_mapping[(i,j)] = mapped_key
    #         if i % 500 == 0:
    #             print_INFO_message("{} out of {}".format(i, len(sp_free_flow)))

    #     adj_matrix_path = retrieve_adj_matrix_path(time, free_flow=True)
    #     adj_mapping_path_2 = retrieve_adj_mapping_path_2(time, free_flow=True)
    #     dataset_adj_matrix = PickleDataSet(adj_matrix_path)
    #     dataset_adj_mapping_2 = PickleDataSet(adj_mapping_path_2)
    #     dataset_adj_matrix.save(adj_matrix)
    #     dataset_adj_mapping_2.save(adj_matrix_mapping)
    
    sp = dict(nx.all_pairs_dijkstra_path_length(average_graph))
    adj_matrix = np.zeros((len(sp), len(sp)))
    adj_matrix_mapping = {}

    print_INFO_message_timestamp("Creating distance matrix")
    for i in range(len((sp))):
        if mapping_points[i] in coordinates_sampled2.geometry:
            for j in range(len(sp)):
                if mapping_points[j] in coordinates_sampled.geometry:
                    min_dis, mapped_key = get_min_distance(i, j, geodf, sp, mapping)
                    adj_matrix[i,j] = min_dis
                    adj_matrix_mapping[(i,j)] = mapped_key
        if i % 500 == 0:
            print_INFO_message("{} out of {}".format(i, len(sp)))

    adj_matrix_path = retrieve_adj_matrix_path(time)
    adj_mapping_path = retrieve_adj_mapping_path(time)
    adj_mapping_path_2 = retrieve_adj_mapping_path_2(time)

    dataset_adj_matrix = PickleDataSet(adj_matrix_path)
    dataset_adj_mapping = PickleDataSet(adj_mapping_path)
    dataset_adj_mapping_2 = PickleDataSet(adj_mapping_path_2)

    dataset_adj_matrix.save(adj_matrix)
    dataset_adj_mapping.save(mapping)
    dataset_adj_mapping_2.save(adj_matrix_mapping)

    finished = True

    return finished


##################################################################### STEP 2 #####################################################################
# ---------------------------------------------- update_data_catalog ---------------------------------------
def update_data_catalog(time, trigger):
    if trigger:
        catalog_path = retrieve_catalog_path()

        with open(catalog_path, "r+") as f:
            contents = f.read()
            adj_path = retrieve_adj_matrix_path(time)
            result = re.search(
                rf"build_adjacency_matrix.{time}.adj_matrix.{time}:", contents
            )
            if result is None:
                contents = "\n".join(
                    [
                        contents,
                        "\n    ".join(
                            [
                                f"build_adjacency_matrix.{time}.adj_matrix.{time}:",
                                f"type: pickle.PickleDataSet",
                                f"filepath: {adj_path}",
                            ]
                        ),
                    ]
                )

            f.seek(0)
            f.truncate()
            f.write(contents)

        with open(catalog_path, "r+") as f:
            contents = f.read()
            adj_mapping_path = retrieve_adj_mapping_path(time)
            result = re.search(
                rf"build_adjacency_matrix.{time}.adj_mapping_matrix.{time}:", contents
            )
            if result is None:
                contents = "\n".join(
                    [
                        contents,
                        "\n    ".join(
                            [
                                f"build_adjacency_matrix.{time}.adj_mapping_matrix.{time}:",
                                f"type: pickle.PickleDataSet",
                                f"filepath: {adj_mapping_path}",
                            ]
                        ),
                    ]
                )

            f.seek(0)
            f.truncate()
            f.write(contents)
