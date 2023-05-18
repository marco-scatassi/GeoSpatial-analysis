import sys
sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')

import warnings
from shapely.errors import ShapelyDeprecationWarning
# Ignore the ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import copy
import pickle
import regex as re
import pandas as pd
import pickle as pkl
import datetime as dt
import networkx as nx
from convert_geometry import toExtremePoints
from kedro.extras.datasets.pickle import PickleDataSet
from log import print_INFO_message_timestamp, print_INFO_message
from retrieve_global_parameters import retrieve_average_graph_path, retrieve_catalog_path, retrieve_gdf_average_path


NEW_NODES = [
    [(5.3184, 60.38643), (5.31856, 60.3851), {"distance": 240}],
    [(5.31868, 60.38503), (5.3184, 60.38643), {"distance": 300}],
    [(5.31619, 60.38728), (5.31702, 60.3862), {"distance": 144}],
    [(5.31726, 60.38627), (5.31645, 60.38717), {"distance": 130}],
    [(5.31726, 60.38627), (5.31619, 60.38728), {"distance": 150}],
    [(5.31565, 60.38753), (5.31702, 60.3862), {"distance": 170}],
    [(5.31762, 60.39184), (5.31522, 60.39234), {"distance": 140}],
    [(5.32031, 60.39494), (5.32107, 60.39467), {"distance": 57}],
    [(5.32031, 60.39494), (5.32056, 60.39432), {"distance": 95}],
    [(5.32513, 60.39416), (5.32031, 60.39494), {"distance": 319}],
    [(5.32513, 60.39416), (5.31995, 60.39523), {"distance": 312}]
]


##################################################################### STEP 1 #####################################################################

def build_graph(gdf_average):
    G = nx.DiGraph()
    
    mapping = {}
    for j, col in enumerate(gdf_average.columns):
            mapping[col] = j
    
    for row in gdf_average.values:
        lines = pd.Series(row[mapping["geometry.multi_line"]])
        speed = row[mapping["currentFlow.speedUncapped"]]
        free_flow_speed = row[mapping["currentFlow.freeFlow"]]
        lengths = row[mapping["geometry_length"]]
        description = row[mapping["description"]]
    
        for i, line in enumerate(lines):
            p0 = toExtremePoints(line)[0]
            p1 = toExtremePoints(line)[-1]
            t0 = tuple([p0.coords.xy[0][0], p0.coords.xy[1][0]])
            t1 = tuple([p1.coords.xy[0][0], p1.coords.xy[1][0]])
        # if the road is closed, set the weight to a very high number
            if speed == 0:
                G.add_edge(t0, t1, weight=100000, speed=speed, free_flow_speed=free_flow_speed, description=description)
            else:
                G.add_edge(t0, t1, weight=lengths[i]/speed, weight2=lengths[i]/free_flow_speed, speed=speed, free_flow_speed=free_flow_speed, description=description)
    return G


def build_strongly_cc(G):
    return [G.subgraph(c).copy() for c in sorted(nx.strongly_connected_components(G), reverse=True, key=len)]


def connect_graph_components(G, CC):
    print_INFO_message_timestamp(f"Nearest nodes to the new nodes:")
    G = copy.deepcopy(G)
    for i, node in enumerate(NEW_NODES):
        for cc in CC:
            if cc.has_node(node[0]):
                if node[0] == (5.31565, 60.38753):
                    predecessor = cc.predecessors(node[0])
                    predecessor = list(predecessor)[1]
                else:
                    predecessor = cc.predecessors(node[0])
                    predecessor = list(predecessor)[0]

                speed1 = cc[predecessor][node[0]]["speed"]
            
            # print the nearest node
                print_INFO_message(f"predecessor of {node[0]} is {predecessor}")
            
            if cc.has_node(node[1]):
                if node[1] == (5.31645, 60.38717) or node[1] == (5.31995, 60.39523):
                    successor = cc.successors(node[1])
                    successor = list(successor)[1]
                else:
                    successor = cc.successors(node[1])
                    successor = list(successor)[0]
            
                speed2 = cc[node[1]][successor]["speed"]
            
            # print the nearest node
                print_INFO_message(f"successor of {node[1]} is {successor}")
            
        avg_speed = (speed1 + speed2) / 2
        weight = node[2]["distance"]/avg_speed
        print_INFO_message(f"average speed is {avg_speed}")
        print_INFO_message(f"weight is {weight}")
  
        G.add_edge(node[0], node[1], weight=weight, speed=avg_speed, distance=node[2]["distance"])
        print_INFO_message(f"added edge between {node[0]} and {node[1]}")
    return G


def build_average_graph(time):
    finished = False
    
    with open(retrieve_gdf_average_path(time), "rb") as f:
        average_gdf = pickle.load(f)
    
    G = build_graph(average_gdf)
    CC = build_strongly_cc(G)
     
    print_INFO_message_timestamp("strongly connected components")
    print_INFO_message(f"{nx.number_strongly_connected_components(G)}")
    
    print_INFO_message_timestamp("nodes number of the largest 4 components")
    print_INFO_message(f"{[c.number_of_nodes() for c in CC[:4]]}")
    
    G_connected = connect_graph_components(G, CC)
    CC_connected = build_strongly_cc(G_connected)
    
    print_INFO_message_timestamp("strongly connected components")
    print_INFO_message(f"{nx.number_strongly_connected_components(G_connected)}")
    
    print_INFO_message_timestamp("nodes number of the largest 4 components")
    print_INFO_message(f"{[c.number_of_nodes() for c in CC_connected[:4]]}")
    
    connected_graph = CC_connected[0]
    
    saving_path = retrieve_average_graph_path(time)
    dataset = PickleDataSet(saving_path)
    dataset.save(connected_graph)
    finished = True
    
    return finished
    

##################################################################### STEP 2 #####################################################################
# ---------------------------------------------- update_data_catalog --------------------------------------- 
def update_data_catalog(time, trigger):
    finished = False
    if trigger:
        catalog_path = retrieve_catalog_path()
        graph_path = retrieve_average_graph_path(time)
        
        with open(catalog_path, "r+") as f:
            contents = f.read()
            result = re.search(fr"average_graph_{time}:", contents)
            if result is None:
                contents = "\n".join([contents, 
                                  "\n    ".join([f"build_average_graphs.{time}.average_graph.{time}:",
                                                 f"type: pickle.PickleDataSet",
                                                 f"filepath: {graph_path}"])])
                
            f.seek(0)
            f.truncate()
            f.write(contents)

    return finished