import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from log import print_INFO_message, print_INFO_message_timestamp
from shapely.geometry import Point
from functools import partial
import time as t
from copy import deepcopy
from graphical_analysis import show_graph

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
        # print_INFO_message(f"at least one of only_predecessors and only_successors must be False")
        raise ValueError
    
    if not only_successors:
        if G.has_node(node[0]):
            predecessors = list(G.predecessors(node[0]))
            # print_INFO_message(f"predecessors of {node[0]} is {predecessors}")
            for predecessor in predecessors:
                edge = (predecessor, node[0])
                speeds.append(list(G.edges(edge, data=True))[0][2]["speed"])
                free_flow_speeds.append(list(G.edges(edge, data=True))[0][2]["free_flow_speed"])
    
    if not only_predecessors:
        if G.has_node(node[1]):
            successors = list(G.successors(node[1]))
            # print_INFO_message(f"successors of {node[1]} is {successors}")
            for successor in successors:
                edge = (node[1], successor)
                speeds.append(list(G.edges(edge, data=True))[1][2]["speed"])
                free_flow_speeds.append(list(G.edges(edge, data=True))[1][2]["free_flow_speed"])
    
    avg_speed = np.mean(speeds)
    avg_free_flow_speed = np.mean(free_flow_speeds)
    weight = node[2] / avg_speed
    weight2 = node[2] / avg_free_flow_speed
    # print_INFO_message(f"average speed is {avg_speed}")
    # print_INFO_message(f"weight is {weight}")
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
            m_n = 0
            m_d = 0
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
            new_path.append(np.round(p0+t0*(p1-p0),6))
        elif n == 0 and not traslate_first_node:
            new_path.append(path[n])
        new_path.append(np.round(p0+t1*(p1-p0),6))
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
    new_node_list = deepcopy(node_list)
    for node in node_list:
        new_node_list += list(G.predecessors(node))
        new_node_list += list(G.successors(node))
        
    for node in new_node_list: 
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

def node_mapping_log(G, node, c=True):
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
    
    nodes = deepcopy(list(node_mapping.keys()))
    for n in nodes:
        ss = list(G.successors(n))
        ps = list(G.predecessors(n))
        for s in ss:
            if s not in node_mapping.keys():
                node_mapping[s] = i
                i += 1
        for p in ps:
            if p not in node_mapping.keys():
                node_mapping[p] = i
                i += 1

    nodes = deepcopy(list(node_mapping.keys()))
    for n in nodes:
        ss = list(G.successors(n))
        ps = list(G.predecessors(n))
        for s in ss:
            if s not in node_mapping.keys():
                node_mapping[s] = i
                i += 1
        for p in ps:
            if p not in node_mapping.keys():
                node_mapping[p] = i
                i += 1
    
    return node_mapping, node_class
        
def update_split_the_node_input(session_state, node, node_mapping, predecessors_id, successors_id):
    session_state["node"] = node
    session_state["node_mapping"] = node_mapping
    session_state["node_mapping_r"] = {v: k for k, v in node_mapping.items()}
            
    session_state["predecessors_id"] = predecessors_id
    session_state["successors_id"] = successors_id

def on_submit_split_the_node_form(session_state, G, node, node_class, img_path, LOG_FILE_PATH2):
    node_mapping = session_state["node_mapping"]
    node_mapping_r = session_state["node_mapping_r"]
    key = str(node)
    
    if key not in session_state["history_changes"].keys():
        session_state["history_changes"][key] = {}
        
    if session_state[f"split_the_node_radio_{node}"] == "yes":
        session_state["history_changes"][key]["split_the_node"] = True
        session_state["history_changes"][key]['selected_predecessor'] = node_mapping_r[session_state[f"predecessor_select_box_{node}"]]
        session_state["history_changes"][key]['selected_successor'] = node_mapping_r[session_state[f"successor_select_box_{node}"]]
        
        new_node_mapping, new_edge = split_the_node_func(G, session_state["history_changes"], node, node_mapping)
        session_state["history_changes"][key]["node_added"] = new_edge[0]
        fig = img_log(G, [node, new_edge[0]], new_node_mapping, node_class)
        fig.write_html(img_path, full_html=True, auto_open=False)
    else:
        session_state["history_changes"][key]["split_the_node"] = False
        session_state["history_changes"][key]['selected_predecessor'] = None
        session_state["history_changes"][key]['selected_successor'] = None
    
    #print_INFO_message_timestamp(f'split the node {node}? (y/n): {session_state["history_changes"][key]["split_the_node"]}', LOG_FILE_PATH2)
    #print_INFO_message(f'selected predecessor: {session_state["history_changes"][key]["selected_predecessor"]}', LOG_FILE_PATH2)
    #print_INFO_message(f'selected successor: {session_state["history_changes"][key]["selected_successor"]}', LOG_FILE_PATH2)
    
    session_state["is_form1_disabled"] = True
    session_state["is_form2_disabled"] = False
              
def split_the_node_input(node, G, node_mapping, node_class, session_state, split_the_node_form_placeholder, img_path, LOG_FILE_PATH2):
    predecessors = list(G.predecessors(node))
    successors = list(G.successors(node))
    
    predecessors_id = [node_mapping[p] for p in predecessors]
    successors_id = [node_mapping[s] for s in successors]
    
    update_split_the_node_input(session_state, node, node_mapping, predecessors_id, successors_id)
    
    split_the_node_form_placeholder.empty()
    split_the_node_form = split_the_node_form_placeholder.form(key=f"split_the_node_form_{node}")
    
    with split_the_node_form:
        node = session_state["node"]
        
        st.write("**Form 1**: split the node "+f'{node}? If "yes", which predecessor and successor?')
                
        st.radio("split the node?", ("yes", "no"), 
                 disabled=session_state["is_form1_disabled"],
                 horizontal=True, 
                 key=f"split_the_node_radio_{node}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("predecessor", 
                        session_state["predecessors_id"],
                        disabled=session_state["is_form1_disabled"], 
                        key = f"predecessor_select_box_{node}",)
        with col2:
            st.selectbox("successor", 
                         session_state["successors_id"], 
                         disabled=session_state["is_form1_disabled"],
                         key = f"successor_select_box_{node}",)
                 
        submit = st.form_submit_button("submit", 
                                       disabled=session_state["is_form1_disabled"],
                                       on_click=on_submit_split_the_node_form, 
                                       args=(session_state, G, node, node_class, img_path, LOG_FILE_PATH2))
      
def on_submit_add_and_delete_edges_form(session_state, G, node, node_mapping_r, img_path=None, log_file_path=None):
    edges_to_add = session_state[f"edges_to_add_{node}"]
    distances_to_add = session_state[f"distances_to_add_{node}"]
    edges_to_delete = session_state[f"edges_to_delete_{node}"]
    
    key = str(node)
    
    dist = distances_to_add.replace(" ", "").split(",")
    if len(edges_to_add) != len(dist) and dist != [""]:
        st.error(f"the number of edges to add and the number of distances provided are different\n{edges_to_add}\n{dist}")
        

    if len(edges_to_add) > 0:
        session_state["history_changes"][key]["new_edges"] = [(node_mapping_r[e[0]], node_mapping_r[e[1]], int(d)) for e, d  in zip(edges_to_add, dist)]
    else:
        session_state["history_changes"][key]["new_edges"] = []
    if len(edges_to_delete) > 0:
        session_state["history_changes"][key]["edges_to_delete"] = [(node_mapping_r[e[0]], node_mapping_r[e[1]]) for e in edges_to_delete]
    else:
        session_state["history_changes"][key]["edges_to_delete"] = []
    
    for e in session_state["history_changes"][key]['new_edges']:
        add_edge(e, G)
    for e in session_state["history_changes"][key]['edges_to_delete']:
        G.remove_edge(e[0], e[1])
    
    if img_path is not None:
        node_mapping, node_class = node_mapping_log(G, node) 
        if "node_added" in session_state["history_changes"][key]:
            node2 = session_state["history_changes"][key]["node_added"]
            fig = img_log(G, [node, node2], node_mapping, node_class)
        else:
            fig = img_log(G, [node], node_mapping, node_class)
        fig.write_html(img_path, full_html=True, auto_open=False)
        
    #print_INFO_message_timestamp(f'new edges: {session_state["history_changes"][key]["new_edges"]}', log_file_path)
    #print_INFO_message_timestamp(f'edges to delete: {session_state["history_changes"][key]["edges_to_delete"]}', log_file_path)
    
    session_state["is_form2_disabled"] = True
    session_state["is_form1_disabled"] = False

def add_and_deleted_edges_input(G, node, session_state, node_mapping, 
                                add_and_delete_form_placeholder,   
                                img_path=None, log_file_path=None):
    node_mapping_r = {v: k for k, v in node_mapping.items()}
    edge_list_add = []
    edge_list_delete = []
    for node1 in node_mapping.keys():
        for node2 in node_mapping.keys():
            if G.has_edge(node1, node2):
                edge_list_delete.append((node_mapping[node1], node_mapping[node2]))
            else:
                edge_list_add.append((node_mapping[node1], node_mapping[node2]))
        
    add_and_delete_form_placeholder.empty()
    add_and_delete_form = add_and_delete_form_placeholder.form(key=f"add_and_delete_form_{node}")
    with add_and_delete_form:
        st.write(f"**Form 2**: add and delete edges for node {node_mapping[node]}")
        st.multiselect("edges to add", 
                       edge_list_add, 
                       disabled=session_state["is_form2_disabled"],
                       key=f"edges_to_add_{node}")
        st.text_input("in order, for each edge added, provide its lenght (m)",
                      disabled=session_state["is_form2_disabled"],
                      placeholder="d1, d2, d3, ...", 
                      key=f"distances_to_add_{node}")
        st.multiselect("edges to delete", 
                       edge_list_delete, 
                       disabled=session_state["is_form2_disabled"],
                       key=f"edges_to_delete_{node}")
        st.form_submit_button("submit", 
                              disabled=session_state["is_form2_disabled"],
                              on_click=on_submit_add_and_delete_edges_form,
                              args=(session_state, G, node, node_mapping_r, img_path, log_file_path))
      
def reconnect_predecessors(G, origin, log_file_path, node, new_edge):
    #print_INFO_message(f"replacing edge {origin}-{node}", log_file_path)
    predecessors = list(G.predecessors(origin))
    for p in predecessors:
        for e in G.edges((p, origin), data=True):
            if e[0] == p and e[1] == origin:
                if nx.has_path(G, p, node) and p!=node:                  
                    #print_INFO_message(f"processing predecessor: {p}", log_file_path)
                    G.remove_edge(p, origin)
                    G.add_edge(p, new_edge[0], **e[2])
                    break
                
def split_the_node_func(G, session_state, node, node_mapping):
    key = str(node)
    selected_predecessor = session_state[key]['selected_predecessor']
    selected_successor = session_state[key]['selected_successor']
    new_edge = traslate_path([(node[0], node[1]), (selected_successor[0], selected_successor[1])], 0.00007, True)
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
    
    return node_mapping, new_edge
            
def split_two_way_roads(G_dict, origin, session_state,
                        split_the_node_form_placeholder,
                        add_and_delete_form_placeholder,
                        count=0, count_max=1, 
                        log_file_path=None, log_file_path2 = None, 
                        clear_log_file=True, img_path=None, G_original=None):

    if type(G_dict) != dict:
        G_dict = {0: G_dict}
    
    G_dict_keys = list(G_dict.keys())
    G = G_dict[G_dict_keys[0]]
    
    if G_original == None:
        G_original = deepcopy(G)
    r=True
    while r:
        if clear_log_file:
            clear_log(log_file_path)
            clear_log(log_file_path2)
                
        #print_INFO_message_timestamp(f"count: {count}", log_file_path)    
        if count > count_max:
            return False
            
        successors = list(G.successors(origin))
        if G_original.has_node(origin):
            original_successors = list(G_original.successors(origin))
        else:
            original_successors = successors
        # print_INFO_message_timestamp(f"origin: {origin}", log_file_path)
        for i, node in enumerate(successors):
            # print_INFO_message(f"succerssors number: {i}", log_file_path)
            only_double = True
            only_single = True
            for s in original_successors:
                if not G_original.has_edge(s, origin):
                    only_double = False
                if G_original.has_edge(s, origin):
                    only_single = False
            check_manually = (not only_double) and (not only_single) 
            if G.has_edge(node, origin):
                #print_INFO_message(f"TWO WAY STREET FOUND", log_file_path)
                new_edge = traslate_path([(origin[0], origin[1]), (node[0], node[1])], 0.00005, True)
                for e in G.edges((origin, node), data=True):
                    if e[0] == origin and e[1] == node:   
                        key = str(node)
                        for H in G_dict.values():
                            reconnect_predecessors(H, origin, log_file_path, node, new_edge)
                                
                            #print_INFO_message(f"old edge is {(origin, node)}", log_file_path)
                            #print_INFO_message(f"new edge is {(new_edge[0], node)}", log_file_path)   
                            H.remove_edge(origin, node) 
                            H.add_edge(new_edge[0], node, **e[2])
                                
                        successors, no_double_sense = check_double_sense_continues(G, node)
                        predecessors = list(G.predecessors(node))
                        is_crossroad_1 = len(successors) > 2
                        is_crossroad_2 =  len(successors) >= 2 and len(predecessors) >= 2
                        for s in successors:
                            for p in predecessors:
                                if s == p:
                                    is_crossroad_2 = False
                        is_crossroad = is_crossroad_1 or is_crossroad_2
                        #print_INFO_message_timestamp(f"no_double_sense: {no_double_sense}", log_file_path)
                            
                        resume_processing = False
                        if key not in session_state.keys():
                            resume_processing = False
                        else:
                            if "new_edges" not in session_state["history_changes"][key].keys():
                                resume_processing = True
                            
                        if check_manually or is_crossroad or resume_processing: 
                            node_mapping, node_class = node_mapping_log(G, node) 
                            fig = img_log(G, [node], node_mapping, node_class)
                            fig.write_html(img_path, full_html=True, auto_open=False)
                                
                            if key in session_state["history_changes"].keys() and \
                                ("split_the_node" in session_state["history_changes"][key].keys() 
                                    and "selected_predecessor" in session_state["history_changes"][key].keys() 
                                        and "selected_successor" in session_state["history_changes"][key].keys()):
                                if session_state["history_changes"][key]['split_the_node']:
                                    for H in G_dict.values():
                                        node_mapping, new_edge = split_the_node_func(H, session_state["history_changes"], node, node_mapping)
                                    fig = img_log(G, [node, new_edge[0]], node_mapping, node_class)
                                    fig.write_html(img_path, full_html=True, auto_open=False)
                                    
                            else:
                                split_the_node_input(node, G, 
                                                     node_mapping, 
                                                     node_class,
                                                     session_state, 
                                                     split_the_node_form_placeholder,
                                                     img_path,
                                                     log_file_path2)
                                
                                while True:
                                    t.sleep(1)
                                        
                            if key in session_state["history_changes"].keys() and \
                                ("new_edges" in session_state["history_changes"][key].keys() and \
                                 "edges_to_delete" in session_state["history_changes"][key].keys()):
                                    for H in G_dict.values():
                                        for e in session_state["history_changes"][key]['new_edges']:
                                            add_edge(e, H)
                                        for e in session_state["history_changes"][key]['edges_to_delete']:
                                            H.remove_edge(e[0], e[1])
                            else:
                                add_and_deleted_edges_input(G, node, session_state, 
                                                            node_mapping, 
                                                            add_and_delete_form_placeholder,
                                                            img_path,
                                                            log_file_path2)
    
                                while True:
                                    t.sleep(1)
                                
                        break
    
            r = split_two_way_roads(G_dict, node, session_state,
                                    split_the_node_form_placeholder,
                                    add_and_delete_form_placeholder,
                                    count+1, count_max, 
                                    log_file_path=log_file_path, 
                                    log_file_path2=log_file_path2,
                                    clear_log_file=False, 
                                    img_path=img_path,
                                    G_original=G_original
                                    )

        return False
            
               
def on_submit_refine_form(session_state, G, node_mapping_r):
    edges_to_add_input = session_state[f"edges_to_add"]
    distances_to_add_input = session_state[f"distances_to_add"]
    edges_to_delete_input = session_state[f"edges_to_delete"]
    
    dist = distances_to_add_input.replace(" ", "").split(",")
    # add = (edges_to_add_input.replace(" ", "").replace(")", "))")+",").split("),")[:-1]
    add = list(eval((edges_to_add_input)))
    delete = (edges_to_delete_input.replace(" ", "").replace(")", "))")+",").split("),")[:-1]
    
    # add = [eval(e.replace("))", ")")) for e in add]
    delete = [eval(e) for e in delete]
    
    if len(add) != len(dist) and dist != [""]:
        st.error(f"the number of edges to add and the number of distances provided are different\n{add}\n{dist}")
    
    if len(add) > 0:
        # new_edges = [(node_mapping_r[e[0]], node_mapping_r[e[1]], int(d)) for e, d  in zip(add, dist)]
        new_edges = []
        for e, d  in zip(add, dist):
            n0 = e[0]
            n1 = e[1]
            if type(e[0]) == int:
                n0 = node_mapping_r[e[0]]
            if type(e[1]) == int:
                n1 = node_mapping_r[e[1]]           
            new_edges.append((n0, n1, int(d)))
    else:
        new_edges = []
    if len(delete) > 0:
        deleted_edges = [(node_mapping_r[e[0]], node_mapping_r[e[1]]) for e in delete]
    else:
        deleted_edges = []

    if new_edges != []:
        session_state["history_changes"]["graph"]["new_edges"] += new_edges
    if deleted_edges != []:
        session_state["history_changes"]["graph"]["edges_to_delete"] += deleted_edges
        
    for e in new_edges:
        add_edge(e, G)
    for e in deleted_edges:
        G.remove_edge(e[0], e[1])

    CCs = build_cc(G, strong=True)
    CCs_ = [G]+CCs[1:]
    fig, _ = show_graph(CCs_)

    session_state["refine_graph"]["is_submitted"] = True
    session_state["refine_graph"]["G"] = G
    session_state["refine_graph"]["fig"] = fig
            
            
def refine_graph(G_dict, form_placeholder, session_state, apply_to_all=False):    
    if type(G_dict) != dict:
        G_dict = {0: G_dict}
    
    G_dict_keys = list(G_dict.keys())
    G = G_dict[G_dict_keys[0]]
    
    node_mapping = {}
    i = 0
    for node in G.nodes():
        node_mapping[node] = i
        i += 1
    
    node_mapping_r = {v: k for k, v in node_mapping.items()}
    
    if "graph" not in session_state["history_changes_refine"].keys():
        session_state["history_changes_refine"]["graph"] = {}
        session_state["history_changes_refine"]["graph"]["new_edges"] = []
        session_state["history_changes_refine"]["graph"]["edges_to_delete"] = []

    if not apply_to_all:
        form_placeholder.empty()
        refine_form = form_placeholder.form(key=f"refine_form")
        with refine_form:
            st.write(f"**Form**: add and delete edges")
            st.text_input("edges to add", 
                        placeholder="(node1, node2), (node3, node4), ...",
                        key=f"edges_to_add")
            st.text_input("in order, for each edge added, provide its lenght (m)",
                        placeholder="d1, d2, d3, ...", 
                        key=f"distances_to_add")
            st.text_input("edges to delete", 
                        placeholder="(node1, node2), (node3, node4), ...",
                        key=f"edges_to_delete")
            st.form_submit_button("submit", 
                                on_click=on_submit_refine_form,
                                args=(session_state, G, node_mapping_r))
    else:
        for H in G_dict.values():
            for e in session_state["history_changes_refine"]["graph"]["new_edges"]:
                add_edge(e, H)
            for e in session_state["history_changes_refine"]["graph"]["edges_to_delete"]:
                H.remove_edge(e[0], e[1])
      
      
