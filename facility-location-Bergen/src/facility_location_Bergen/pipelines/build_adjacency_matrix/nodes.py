import sys

sys.path.append(
    r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
)

import warnings
from shapely.errors import ShapelyDeprecationWarning

# Ignore the ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import copy
import pickle
import numpy as np
import regex as re
import pandas as pd
import pickle as pkl
import datetime as dt
import networkx as nx
from kedro.extras.datasets.pickle import PickleDataSet
from convert_geometry import toMultiLineString, toExtremePoints
from log import print_INFO_message_timestamp, print_INFO_message
from retrieve_global_parameters import (
    retrieve_average_graph_path,
    retrieve_catalog_path,
    retrieve_adj_matrix_path,
    retrieve_adj_mapping_path,
)


##################################################################### STEP 1 #####################################################################


def build_adj_matrix(time):
    finished = False
    average_graph_path = retrieve_average_graph_path(time, True, True, True, False)

    with open(average_graph_path, "rb") as f:
        average_graph = pickle.load(f)

    mapping = {}
    for i, node in enumerate(average_graph.nodes()):
        mapping[i] = node

    # if time == "all_day":
    #     sp_free_flow = dict(
    #         nx.all_pairs_dijkstra_path_length(average_graph, weight="weight2")
    #     )
    #     adj_matrix = np.zeros((len(sp_free_flow), len(sp_free_flow)))

    #     print_INFO_message_timestamp("Creating free_flow distance matrix")
    #     for i in range(len((sp_free_flow))):
    #         for j in range(len(sp_free_flow)):
    #             adj_matrix[i, j] = sp_free_flow[mapping[i]][mapping[j]]
    #         if i % 500 == 0:
    #             print_INFO_message("{} out of {}".format(i, len(sp_free_flow)))

    #     adj_matrix_path = retrieve_adj_matrix_path(time, free_flow=True)
    #     dataset_adj_matrix = PickleDataSet(adj_matrix_path)
    #     dataset_adj_matrix.save(adj_matrix)
    
    sp = dict(nx.all_pairs_dijkstra_path_length(average_graph))
    adj_matrix = np.zeros((len(sp), len(sp)))

    print_INFO_message_timestamp("Creating distance matrix")
    for i in range(len((sp))):
        for j in range(len(sp)):
            adj_matrix[i, j] = sp[mapping[i]][mapping[j]]
        if i % 500 == 0:
            print_INFO_message("{} out of {}".format(i, len(sp)))

    adj_matrix_path = retrieve_adj_matrix_path(time)
    adj_mapping_path = retrieve_adj_mapping_path(time)

    dataset_adj_matrix = PickleDataSet(adj_matrix_path)
    dataset_adj_mapping = PickleDataSet(adj_mapping_path)

    dataset_adj_matrix.save(adj_matrix)
    dataset_adj_mapping.save(mapping)

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
