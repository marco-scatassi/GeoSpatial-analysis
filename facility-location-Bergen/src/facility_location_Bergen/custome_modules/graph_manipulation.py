import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from log import print_INFO_message, print_INFO_message_timestamp
from shapely.geometry import LineString, Point, Polygon


def build_cc(G, strong=False):
    if not strong:
        if nx.is_directed(G):
            G = G.to_undirected()
        CCs = [nx.DiGraph(G.subgraph(c).copy()) for c in sorted(nx.connected_components(G), reverse=True, key=len)]
    else:
        CCs = [G.subgraph(c).copy() for c in sorted(nx.strongly_connected_components(G), reverse=True, key=len)]
    return CCs

def add_edge(node, G, only_successors=False, only_predecessors=False):
    speeds = []
    free_flow_speeds = []

    if only_predecessors and only_successors:
        print_INFO_message(f"at least one of only_predecessors and only_successors must be False")
        raise ValueError
    
    if not only_successors:
        predecessors = list(G.predecessors(node[0]))
        print_INFO_message(f"predecessors of {node[0]} is {predecessors}")
        for predecessor in predecessors:
            edge = (predecessor, node[0])
            speeds.append(list(G.edges(edge, data=True))[0][2]["speed"])
            free_flow_speeds.append(list(G.edges(edge, data=True))[0][2]["free_flow_speed"])
    
    if not only_predecessors:
        if G.has_node(node[1]):
            successors = list(G.successors(node[1]))
            print_INFO_message(f"successors of {node[1]} is {successors}")
            for successor in successors:
                edge = (node[1], successor)
                speeds.append(list(G.edges(edge, data=True))[1][2]["speed"])
                free_flow_speeds.append(list(G.edges(edge, data=True))[1][2]["free_flow_speed"])
    
    avg_speed = np.mean(speeds)
    avg_free_flow_speed = np.mean(free_flow_speeds)
    weight = node[2] / avg_speed
    weight2 = node[2] / avg_free_flow_speed
    print_INFO_message(f"average speed is {avg_speed}")
    print_INFO_message(f"weight is {weight}")
    G.add_edge(node[0],
                node[1],
                weight=weight,
                weight2=weight2,
                speed=avg_speed,
                free_flow_speed=avg_free_flow_speed)

def traslate_path(path, factor=0.01, traslate_first_node=False):
    new_path = []
    for n in range(len(path)-1):
        if path[n+1][0]-path[n][0] == 0:
            m = 0
        else:
            m_n = (path[n+1][1]-path[n][1])
            m_d = (path[n+1][0]-path[n][0])
        if  (m_n<0 and m_d<0) or (m_d<0 and m_n>0):
            p0 = np.array((path[n][0], path[n][1]+factor))
            p1 = np.array((path[n+1][0], path[n+1][1]+factor))
        else:
            p0 = np.array((path[n][0], path[n][1]-factor))
            p1 = np.array((path[n+1][0], path[n+1][1]-factor))
        l2 = np.sum((p0-p1)**2)
        t0 = np.sum((np.array(path[n])-p0)*(p1-p0))/l2
        t1 = np.sum((np.array(path[n+1])-p0)*(p1-p0))/l2
        if n == 0 and traslate_first_node:
            new_path.append(np.round(p0+t0*(p1-p0),5))
        elif n == 0 and not traslate_first_node:
            new_path.append(path[n])
        new_path.append(np.round(p0+t1*(p1-p0),5))
    return [tuple(p) for p in new_path]

def clear_log(log_file_path):
    with open(log_file_path, "w") as f:
        f.write("")
        
def check_double_sense_continues(G, node):
    successors = list(G.successors(node))
    no_double_sense = True
    for s in successors:
        if G.has_edge(s, node):
            no_double_sense = False
            break
    return (successors, no_double_sense)

def img_log(G, node_list, node_mapping, node_class=[]):
    fig = go.Figure()
    for node in node_list: 
        nodes_lon = []
        nodes_lat = []
        weights = []
        nodes = []
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))
        for edge in G.edges(data=True):
            if edge[0] == node or edge[1] == node \
            or edge[0] in predecessors or edge[1] in successors \
            or edge[0] in successors or edge[1] in predecessors:       
                x0, y0 = edge[0]
                x1, y1 = edge[1]
                weight = edge[2]["weight"]
                nodes_lon.append(x0)
                nodes_lon.append(x1)
                nodes_lon.append(None)
                nodes_lat.append(y0)
                nodes_lat.append(y1)
                nodes_lat.append(None)
                weights.append(weight)
                nodes.append(edge[0])
                nodes.append(edge[1])
                                    
        fig.add_trace(go.Scattermapbox(
                                                lat=nodes_lat,
                                                lon=nodes_lon,
                                                mode='lines',
                                                line=dict(width=1, color="red"),
                                                showlegend=False,
                                            ))
                                    
        nodes = gpd.GeoDataFrame(
                                pd.Series(nodes).apply(lambda x: Point(x)), 
                                columns=["geometry"], 
                                crs="EPSG:4326")
        text = []
        for n in nodes.geometry:
            text.append("")
            if (n.x, n.y) in node_mapping.keys():
                text[-1] += str(node_mapping[(n.x, n.y)])
            else:
                text[-1] += "-"
            if (n.x, n.y) in node_class.keys():
                text[-1] += f" {str(node_class[(n.x, n.y)])}"
        
        fig.add_trace(go.Scattermapbox(
                                    lat = nodes.geometry.y,
                                    lon = nodes.geometry.x,
                                    mode='markers+text',
                                    marker=dict(size=5, color="black"),
                                    text = text,
                                    showlegend=False,
                                ))  
        
    fig.update_layout(mapbox=dict(
                                style="open-street-map",
                                center=dict(lat=np.mean(nodes.geometry.y), lon=np.mean(nodes.geometry.x)),
                                zoom=16
                                ),
                        height=700,
                        width=600,)

    return fig

def node_mapping_log(G, node):
    node_mapping = {node: 0}
    node_class = {node: ["origin"]}
    i = 1
    for edge in G.edges(data=True):
        if edge[0] == node:
            if edge[1] not in node_mapping.keys():
                node_mapping[edge[1]] = i
                i += 1
            if edge[1] not in node_class.keys():
                node_class[edge[1]] = ["successor"]
            else:
                node_class[edge[1]].append("successor")
        if edge[1] == node:
            if edge[0] not in node_mapping.keys():
                node_mapping[edge[0]] = i
                i += 1
            if edge[0] not in node_class.keys():
                node_class[edge[0]] = ["predecessor"]
            else:
                node_class[edge[0]].append("predecessor")
    return node_mapping, node_class

def split_the_node_input(node, node_mapping, message="split the node "):
    split_the_node = input(message+f"{node_mapping[node]}? (y/n): ")
    if split_the_node == "y":
        return True
    elif split_the_node == "n":
        return False
    else:
        return split_the_node_input(node, node_mapping, "invalid input. split the node ")
    
def select_pre_and_suc_input(node, G, node_mapping, message=""):
    node_mapping_r = {v: k for k, v in node_mapping.items()}
    predecessors = list(G.predecessors(node))
    successors = list(G.successors(node))
    try:
        predecessor, successor = input(message+
                                       "select a predecessor and a successor of "+
                                       f"{node_mapping[node]} (comma separated): ").split(",")
        predecessor = node_mapping_r[int(predecessor)]
        successor = node_mapping_r[int(successor)]
    except:
        return select_pre_and_suc_input(node, G, node_mapping, "invalid input. ")

    if predecessor not in predecessors:
        return select_pre_and_suc_input(node, G, node_mapping, "not a predecessor. ")
    elif successor not in successors:
        return select_pre_and_suc_input(node, G, node_mapping, "not a successor. ")
    else:
        return predecessor, successor
    
def new_edges_input(node_mapping, message="insert new connections (label0, label1, distance(m); ...): "):
    new_edges = input(message)
    node_mapping_r = {v: k for k, v in node_mapping.items()}
    if new_edges == "":
        new_edges = []
    else:
        new_edges = new_edges.split(";")
        if "" in new_edges:
            new_edges.remove("")
        try:
            new_edges = [(int(e.split(",")[0]), int(e.split(",")[1]), int(e.split(",")[2])) for e in new_edges]
            new_edges = [(node_mapping_r[e[0]], node_mapping_r[e[1]], e[2]) for e in new_edges]    
        except:
            new_edges = new_edges_input(node_mapping, "invalid input, try again (label+distance(m) semicolon separated):")
    return new_edges

def edges_to_delete_input(node_mapping, message="insert edges to delete (label0, label1; ...): "):
    edges_to_delete = input(message)
    node_mapping_r = {v: k for k, v in node_mapping.items()}
    if edges_to_delete == "":
        edges_to_delete = []
    else:
        try:
            edges_to_delete = edges_to_delete.split(";")
            if "" in edges_to_delete:
                edges_to_delete.remove("")
            edges_to_delete = [(int(e.split(",")[0]), int(e.split(",")[1])) for e in edges_to_delete]
            edges_to_delete = [(node_mapping_r[e[0]], node_mapping_r[e[1]]) for e in edges_to_delete]
        except:
            edges_to_delete = new_edges_input(node_mapping, "invalid input, try again (tuple label semicolon separated):")
    return edges_to_delete

def reconnect_predecessors(G, origin, log_file_path, node, new_edge):
    print_INFO_message(f"replacing edge {origin}-{node}", log_file_path)
    predecessors = list(G.predecessors(origin))
    for p in predecessors:
        for e in G.edges((p, origin), data=True):
            if e[0] == p and e[1] == origin:
                if nx.has_path(G, p, node) and p!=node:                  
                    print_INFO_message(f"processing predecessor: {p}", log_file_path)
                    G.remove_edge(p, origin)
                    G.add_edge(p, new_edge[0], **e[2])
                    break
                
def split_the_node_func(G, log_file_path2, img_path, history_changes, node, node_mapping, node_class, get_data_from_input=True):
    if get_data_from_input:
        selected_predecessor, selected_successor = select_pre_and_suc_input(node, G, node_mapping)
        history_changes[node]['selected_predecessor'] = selected_predecessor
        history_changes[node]['selected_successor'] = selected_successor
        print_INFO_message(f"selected predecessor: {selected_predecessor}", log_file_path2)
        print_INFO_message(f"selected successor: {selected_successor}", log_file_path2)
        new_edge = traslate_path([(node[0], node[1]), (selected_successor[0], selected_successor[1])], 0.00005, True)
        node_mapping[new_edge[0]] = max(list(node_mapping.values()))+1
        for e in G.edges((selected_predecessor, node), data=True):
            if e[0] == selected_predecessor and e[1] == node:
                G.add_edge(selected_predecessor, new_edge[0], **e[2])
                G.remove_edge(selected_predecessor, node)
                break
        for e in G.edges((node, selected_successor), data=True):
            if e[0] == node and e[1] == selected_successor:
                G.add_edge(new_edge[0], selected_successor, **e[2])
                G.remove_edge(node, selected_successor)
                break
                                        
        fig = img_log(G, [node, new_edge[0]], node_mapping, node_class)
        fig.write_html(img_path, full_html=True, auto_open=True)
    else:
        selected_predecessor = history_changes[node]['selected_predecessor']
        selected_successor = history_changes[node]['selected_successor']
        new_edge = traslate_path([(node[0], node[1]), (selected_successor[0], selected_successor[1])], 0.00005, True)
        node_mapping[new_edge[0]] = max(list(node_mapping.values()))+1
        for e in G.edges((selected_predecessor, node), data=True):
            if e[0] == selected_predecessor and e[1] == node:
                G.add_edge(selected_predecessor, new_edge[0], **e[2])
                G.remove_edge(selected_predecessor, node)
                break
        for e in G.edges((node, selected_successor), data=True):
            if e[0] == node and e[1] == selected_successor:
                G.add_edge(new_edge[0], selected_successor, **e[2])
                G.remove_edge(node, selected_successor)
                break
            
def split_two_way_roads(G, origin, count=0, count_max=1, 
                        log_file_path=None, log_file_path2 = None, clear_log_file=True, img_path=None,
                        history_changes = {},):
    if clear_log_file:
        clear_log(log_file_path)
        clear_log(log_file_path2)
        
    print_INFO_message_timestamp(f"count: {count}", log_file_path)    
    if count > count_max:
        return
    
    successors = list(G.successors(origin))
    # print_INFO_message_timestamp(f"origin: {origin}", log_file_path)
    for i, node in enumerate(successors):
        # print_INFO_message(f"succerssors number: {i}", log_file_path)
        if G.has_edge(node, origin):
            print_INFO_message(f"TWO WAY STREET FOUND", log_file_path)
            new_edge = traslate_path([(origin[0], origin[1]), (node[0], node[1])], 0.00005, True)
            for e in G.edges((origin, node), data=True):
                if e[0] == origin and e[1] == node:
                    reconnect_predecessors(G, origin, log_file_path, node, new_edge)
                    
                    print_INFO_message(f"old edge is {(origin, node)}", log_file_path)
                    print_INFO_message(f"new edge is {(new_edge[0], node)}", log_file_path)   
                    G.remove_edge(origin, node) 
                    G.add_edge(new_edge[0], node, **e[2])
                        
                    successors, no_double_sense = check_double_sense_continues(G, node)
                    is_crossroad = len(successors) > 2
                    print_INFO_message_timestamp(f"no_double_sense: {no_double_sense}", log_file_path)
                    if no_double_sense or is_crossroad: 
                        node_mapping, node_class = node_mapping_log(G, node) 
                        if node in history_changes.keys():
                            for k in ['split_the_node', 'selected_predecessor', 'selected_successor', 'new_edges', 'edges_to_delete']:
                                if k not in history_changes[node].keys():
                                    raise Exception(f"key {k} not in history_changes[{node}]")
                                
                            print_INFO_message_timestamp(f"node {node} already in history_changes")
                            print_INFO_message(history_changes[node])
                            
                            if history_changes[node]['split_the_node']:
                                split_the_node_func(G, log_file_path2, img_path, history_changes, node, node_mapping, node_class, False)
                                   
                            for e in history_changes[node]['new_edges']:
                                add_edge(e, G)
                            
                            for e in history_changes[node]['edges_to_delete']:
                                G.remove_edge(e[0], e[1])
                            
                        else:
                            history_changes[node] = {}
                            fig = img_log(G, [node], node_mapping, node_class)
                            fig.write_html(img_path, full_html=True, auto_open=False)
                            
                            split_the_node = split_the_node_input(node, node_mapping)
                            history_changes[node]['split_the_node'] = split_the_node
                            print_INFO_message_timestamp(f"split the node {node}? (y/n): {split_the_node}", log_file_path2)
                            
                            if split_the_node:
                                while split_the_node:
                                    split_the_node_func(G, log_file_path2, img_path, history_changes, node, node_mapping, node_class)
                                    node_mapping, node_class = node_mapping_log(G, node) 
                                    split_the_node = split_the_node_input(node, node_mapping, "split again the node ")
                    
                            new_edges = new_edges_input(node_mapping)
                            history_changes[node]['new_edges'] = new_edges
                            for e in new_edges:
                                add_edge(e, G)
                            print_INFO_message(f"new edges: {new_edges}", log_file_path2)
                            
                            edges_to_delete = edges_to_delete_input(node_mapping)
                            history_changes[node]['edges_to_delete'] = edges_to_delete
                            for e in edges_to_delete:
                                G.remove_edge(e[0], e[1])
                            print_INFO_message(f"edges to delete: {edges_to_delete}", log_file_path2)

                            fig = img_log(G, [node], node_mapping, node_class)
                            fig.write_html(img_path, full_html=True, auto_open=False)
                            input("press enter to continue")
                    break

        split_two_way_roads(G, node, count+1, count_max, 
                            log_file_path=log_file_path, 
                            log_file_path2=log_file_path2,
                            clear_log_file=False, 
                            img_path=img_path,
                            history_changes=history_changes)
    