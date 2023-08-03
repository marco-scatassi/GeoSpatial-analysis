import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from log import print_INFO_message, print_INFO_message_timestamp
from shapely.geometry import Point
from functools import partial

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
        
def split_the_node_input(node, G, node_mapping, session_state, log_file_path2, message="split the node "):
    stay = True
    session_state[node] = {}
    node_mapping_r = {v: k for k, v in node_mapping.items()}
    predecessors = list(G.predecessors(node))
    successors = list(G.successors(node))
    
    predecessors_id = [node_mapping[p] for p in predecessors]
    successors_id = [node_mapping[s] for s in successors]
    
    with st.form(f"split the node {node}"):
        st.write(message+f'{node_mapping[node]}? If "yes", which predecessor and successor?')
        split_the_node = st.radio("split the node?", ("yes", "no"))
        col1, col2 = st.columns(2)
        with col1:
            slected_predecessor_id = st.selectbox("predecessor", predecessors_id)
        with col2:
            selected_successor_id = st.selectbox("successor", successors_id)
        submit = st.form_submit_button("submit")
        
        while stay: 
            if submit:
                if split_the_node == "yes":
                    session_state[node]["split_the_node"] = True
                    session_state[node]['selected_predecessor'] = node_mapping_r[slected_predecessor_id]
                    session_state[node]['selected_successor'] = node_mapping_r[selected_successor_id]
                else:
                    session_state[node]["split_the_node"] = False
                    session_state[node]['selected_predecessor'] = None
                    session_state[node]['selected_successor'] = None
                
                print_INFO_message_timestamp(f'split the node {node}? (y/n): {session_state[node]["split_the_node"]}', log_file_path2)
                print_INFO_message(f'selected predecessor: {session_state[node]["selected_predecessor"]}', log_file_path2)
                print_INFO_message(f'selected successor: {session_state[node]["selected_successor"]}', log_file_path2)

                stay = False
                      
def add_and_deleted_edges(G, node, session_state, node_mapping, log_file_path):
    stay = True
    node_mapping_r = {v: k for k, v in node_mapping.items()}
    edge_list = [None]
    for node1 in node_mapping.keys():
        for node2 in node_mapping.keys():
            if G.has_edge(node1, node2):
                edge_list.append((node_mapping[node1], node_mapping[node2]))
    
    with st.form(f"add and delete edges for node {node}"):
        st.write(f"add and delete edges for node {node_mapping[node]}")
        edges_to_add = st.multiselect("edges to add", edge_list)
        edges_to_delete = st.multiselect("edges to delete", edge_list)
        submit = st.form_submit_button("submit")
        
        while stay: 
            if submit:
                if None in edges_to_add:
                    if len(edges_to_add) > 1:
                        edges_to_add.remove(None)
                        session_state[node]["new_edges"] = [(node_mapping_r[e[0]], node_mapping_r[e[1]]) for e in edges_to_add]
                    else:
                        session_state[node]["new_edges"] = []
                if None in edges_to_delete:
                    if len(edges_to_delete) > 1:
                        edges_to_delete.remove(None)
                        session_state[node]["edges_to_delete"] = [(node_mapping_r[e[0]], node_mapping_r[e[1]]) for e in edges_to_delete]
                    else:
                        session_state[node]["edges_to_delete"] = []
                        
                print_INFO_message_timestamp(f'new edges: {session_state[node]["new_edges"]}', log_file_path)
                print_INFO_message_timestamp(f'edges to delete: {session_state[node]["edges_to_delete"]}', log_file_path)

                stay = False

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
                
def split_the_node_func(G, session_state, node, node_mapping):
    selected_predecessor = session_state[node]['selected_predecessor']
    selected_successor = session_state[node]['selected_successor']
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
            
def split_two_way_roads(G, origin, session_state,
                        count=0, count_max=1, 
                        log_file_path=None, log_file_path2 = None, clear_log_file=True, img_path=None):
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
                    
                    resume_processing = False
                    if node not in session_state.keys():
                        resume_processing = False
                    else:
                        if "new_edges" not in session_state[node].keys():
                            resume_processing = True
                    
                    if no_double_sense or is_crossroad or resume_processing: 
                        node_mapping, node_class = node_mapping_log(G, node) 
                        fig = img_log(G, [node], node_mapping, node_class)
                        fig.write_html(img_path, full_html=True, auto_open=False)
                        
                        if node in session_state.keys() and ("split_the_node" in session_state[node].keys() 
                                and "selected_predecessor" in session_state[node].keys() 
                                    and "selected_successor" in session_state[node].keys()):
        
                            if session_state[node]['split_the_node']:
                                split_the_node_func(G, session_state, node, node_mapping)
                                fig = img_log(G, [node, new_edge[0]], node_mapping, node_class)
                                fig.write_html(img_path, full_html=True, auto_open=True)
                            
                        else:
                            split_the_node_input(node, G, node_mapping, session_state, log_file_path2)
                                
                        if "new_edges" in session_state[node].keys() and "edges_to_delete" in session_state[node].keys():
                                for e in session_state[node]['new_edges']:
                                    add_edge(e, G)
                                for e in session_state[node]['edges_to_delete']:
                                    G.remove_edge(e[0], e[1])
                        else:
                            add_and_deleted_edges(G, node, session_state, node_mapping, log_file_path2)
                        

                            fig = img_log(G, [node], node_mapping, node_class)
                            fig.write_html(img_path, full_html=True, auto_open=False)
                            input("press enter to continue")

                    break

        split_two_way_roads(G, node, session_state,
                            count+1, count_max, 
                            log_file_path=log_file_path, 
                            log_file_path2=log_file_path2,
                            clear_log_file=False, 
                            img_path=img_path,
                            )
    