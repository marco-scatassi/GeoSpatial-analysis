import sys

# Get the directory path to add to PYTHONPATH
directory_path = r"/mount/src/geospatial-analysis/facility-location-Bergen/src/facility_location_Bergen/custome_modules"
directory_path = r"/app/geospatial-analysis/facility-location-Bergen/src/facility_location_Bergen/custome_modules"
if directory_path not in sys.path:
    sys.path.append(directory_path)
    
import os
import pickle as pkl
import random
import numpy as np
import time as ptime
from copy import deepcopy
from pathlib import Path
import networkx as nx

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from facility_location import (
    FacilityLocation,
    FacilityLocationReport,
    StochasticFacilityLocation,
)
from graph_manipulation import *
from kedro.framework.project import find_pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.pipeline import Pipeline, pipeline
from kedro.runner import SequentialRunner
from log import print_INFO_message, print_INFO_message_timestamp
from PIL import Image
from retrieve_global_parameters import (
    retrieve_average_graph_path,
    retrieve_light_solution_path,
    retrieve_solution_vs_scenario_path,
)
from src.facility_location_Bergen.custome_modules.graphical_analysis import *
from streamlit_folium import st_folium


st.set_page_config(layout = "wide")
session_state = st.session_state

project_path = r"/mount/src/geospatial-analysis/facility-location-Bergen"
project_path = r"/app/geospatial-analysis/facility-location-Bergen"
metadata = bootstrap_project(project_path)

TIMES = ["all_day_free_flow", "all_day", "morning", "midday", "afternoon"]
FACILITIES_NUMBER = [1,2,3]

# LOG_FILE_PATH = r"/mount/src/geospatial-analysis/facility-location-Bergen/logs/split_roads.log"
# LOG_FILE_PATH2 = r"/mount/src/geospatial-analysis/facility-location-Bergen/logs/split_roads_changes.log"
# HTML_IMG_PATH = r"/mount/src/geospatial-analysis/facility-location-Bergen/logs/img_split_roads.html"

LOG_FILE_PATH = r"/app/geospatial-analysis/facility-location-Bergen/logs/split_roads.log"
LOG_FILE_PATH2 = r"/app/geospatial-analysis/facility-location-Bergen/logs/split_roads_changes.log"
HTML_IMG_PATH = r"/app/geospatial-analysis/facility-location-Bergen/logs/img_split_roads.html"

GRAPH_MANIPULATION_SEED=8797
# --------------------------------------------- UTILITY AND CALLBACK --------------------------------------------
def initialize_session_state_attributes(from_graph_button_load=False):
    keys = ["node", "modified_graph", "refine_graph", "history_changes", 
            "history_changes_refine", "load_data_error","node_mapping", "predecessors_id", 
            "successors_id", "apply_graph_modification","stop_and_clear", "button_load", 
            "is_form1_disabled", "is_form2_disabled"]
    
    default = ["___", None, {}, {}, {}, False, {}, [], [], False, False, False, False, True]
    
    for key, value in zip(keys, default):
        if key not in st.session_state:
            st.session_state[key] = value
    
    if from_graph_button_load:
        st.session_state["button_load"] = True
        st.session_state["stop_and_clear"] = False
        st.session_state["load_data_error"] = False
        st.session_state["is_form1_disabled"] = False
        st.session_state["is_form2_disabled"] = True
        st.session_state["apply_graph_modification"] = False
        st.session_state["checkpoint"] = {}
        if "is_submitted" in st.session_state["refine_graph"].keys(): 
            st.session_state["refine_graph"]["is_submitted"] = False
        if "is_submitted2" in st.session_state["refine_graph"].keys():
            st.session_state["refine_graph"]["is_submitted2"] = False
        if st.session_state["upload_button_0"] is not None:
            st.session_state["modified_graph"] = pkl.load(st.session_state["upload_button_0"])
        if st.session_state["upload_button_1"] is not None:
            st.session_state["history_changes"] = pkl.load(st.session_state["upload_button_1"])
        
 
def clear_log_files():
    with open(LOG_FILE_PATH, "w") as f:
        f.write("")
    
    with open(LOG_FILE_PATH2, "w") as f:
        f.write("")
        
    with open(HTML_IMG_PATH, "w") as f:
        f.write("""<!DOCTYPE html>
                <html>
                <head>
                <style>
                h1 {text-align: center;}
                p {text-align: center;}
                div {text-align: center;}
                </style>
                </head>
                
                <body>
                <h1>Graph visualization</h1>
                <p>Click the refresh button to update the image</p>
                </body>
                </html>""")
    
def stop_and_clear_callback():
    keys = ["node", "node_mapping", "predecessors_id", "successors_id", 
            "stop_and_clear", "button_load", "is_form1_disabled", "is_form2_disabled"]
    st.session_state["stop_and_clear"] = True
    st.session_state["button_load"] = False
    st.session_state["refine_graph"]["is_submitted"] = False
    st.session_state["apply_graph_modification"] = False
    st.session_state["load_data_error"] = False
    for key in keys:
        if key != "stop_and_clear" and key != "button_load":
            del st.session_state[key]

def on_submit_refine(placeholder):
    st.session_state["apply_graph_modification"] = False
    st.session_state["stop_and_clear"] = True
    st.session_state["button_load"] = False
    with placeholder:
        for att in ["average_graphs", "node", "node_mapping", "predecessors_id", "successors_id", "stop_and_clear", "button_load"]:
            if att not in st.session_state:
                st.session_state["load_data_error"] = True
                return st.error("Please load data first!", icon="🚨")
    
                

    if "refine_graph" not in session_state:
        session_state["refine_graph"] = {}
                
    G1 = session_state["modified_graph"]
    G2 = session_state["average_graphs"]["all_day"]
    G = G1 if G1 is not None else G2
        
    session_state["n_strongly_cc"] = nx.number_strongly_connected_components(G) 
    CCs = build_cc(G, strong=True)
    CCs_ = [G]+CCs[1:]
    fig, _ = show_graph(CCs_)

    session_state["refine_graph"]["is_submitted"] = True
    session_state["refine_graph"]["G"] = G
    session_state["refine_graph"]["fig"] = fig

def on_submit_apply(placeholder):
    st.session_state["apply_graph_modification"] = True
    session_state["refine_graph"]["is_submitted"] = False
    st.session_state["stop_and_clear"] = True
    st.session_state["button_load"] = False
    with placeholder:
        for att in ["average_graphs", "node", "node_mapping", "predecessors_id", "successors_id", "stop_and_clear", "button_load"]:
            if att not in st.session_state:
                st.session_state["load_data_error"] = True
                return st.error("Please load data first!", icon="🚨")
            
def on_submit_apply2():
    session_state["refine_graph"]["is_submitted2"] = True


# --------------------------------------------- GRAPH MANIPULATION ----------------------------------------------
def graph_manipulation_load_data(session_state, TIMES):
    progress_bar = st.progress(0, "Loading data...")

    if f"average_graphs" not in session_state:
        average_graphs = {}
        for i, time in enumerate(TIMES):
            progress_bar.progress((i+1)*1/len(TIMES), f"Loading average graph for: {time}")
            if time == "all_day_free_flow":
                continue
            path = project_path+"/"+retrieve_average_graph_path(time, connected=True)
            with open(path, "rb") as f:
                average_graphs[time] = pkl.load(f)

        session_state[f"average_graphs"] = average_graphs
    
    session_state["history_changes"] = {}

    progress_bar.progress(100, "Loading data completed!")

def graph_manipulation_process(session_state, LOG_FILE_PATH, LOG_FILE_PATH2, HTML_IMG_PATH, GRAPH_MANIPULATION_SEED, 
                               split_the_node_form_placeholder, add_and_delete_form_placeholder, apply_to_all=False, progress_bar_placeholder=None):

    # if "checkpoint" not in session_state.keys():
    #     session_state["checkpoint"] = {}
    if progress_bar_placeholder is not None:
        progress_bar = progress_bar_placeholder.progress(0, "Splitting two way roads...")

    if apply_to_all:
        session_state["modified_graph"] = {}
        session_state["checkpoint"] = {}
        for key in session_state[f"average_graphs"].keys():
            session_state["modified_graph"][key] = deepcopy(session_state[f"average_graphs"][key])     
            session_state["checkpoint"][key] = {}
    else:
        session_state["modified_graph"] = deepcopy(session_state[f"average_graphs"]["all_day"])            
    
    nodes = list(session_state[f"average_graphs"]["all_day"].nodes())
    seed = random.seed(GRAPH_MANIPULATION_SEED)

    origin = random.choice(nodes)
    print_INFO_message_timestamp("Splitting two way roads")
    for i in range(600):
        print_INFO_message(f"iteration {i}") 
        if progress_bar_placeholder is not None:
            progress_bar.progress((i+1)*1/600, f"Splitting two way roads... {i+1}/600")
            
        if i%20 == 0 and i in session_state["checkpoint"].keys():
            if apply_to_all:
               for key in session_state["modified_graph"].keys():
                   session_state["modified_graph"][key] = session_state["checkpoint"][key][i]
                   
            c_max = -1
        else:
            c_max = 80
        split_two_way_roads(session_state["modified_graph"], 
                                        origin=origin, 
                                        session_state=session_state,
                                        split_the_node_form_placeholder=split_the_node_form_placeholder,
                                        add_and_delete_form_placeholder=add_and_delete_form_placeholder,
                                        count=0,
                                        count_max=c_max, 
                                        log_file_path=LOG_FILE_PATH,
                                        log_file_path2=LOG_FILE_PATH2, 
                                        img_path=HTML_IMG_PATH,)

        if i%20 == 0:
            if apply_to_all:
                for key in session_state["modified_graph"].keys():
                    session_state["checkpoint"][key][i] = deepcopy(session_state["modified_graph"][key])
                else:
                    session_state["checkpoint"][i] = deepcopy(session_state["modified_graph"])
        #     session_state["checkpoint"][i] = deepcopy(session_state["modified_graph"])
        # else:
        #     session_state["modified_graph"] = session_state["checkpoint"][i]
        
        origin = random.choice(nodes)

    
    
def graph_manipulation_process_template(session_state, TIMES, 
                               LOG_FILE_PATH, LOG_FILE_PATH2, HTML_IMG_PATH, GRAPH_MANIPULATION_SEED):
    
    text_col, img_col = st.columns(2)
    with open(HTML_IMG_PATH, "r", encoding="utf-8") as f:
        html_img = f.read()
    
    with img_col:
        st.components.v1.html(html_img, height=600)
        _, refresh_col, _, stop_and_clear_col, _ = st.columns(5)
        with refresh_col:
            st.button("refresh image")
        with stop_and_clear_col:
            stop_and_clear_button = st.button("Stop process and clear memory", 
                                             on_click=stop_and_clear_callback,)
            
    with text_col:
        split_the_node_form_placeholder = st.empty()
        split_the_node_form = split_the_node_form_placeholder.form(f"split the node form")
        with split_the_node_form:
            node = session_state["node"]
                
            st.write("**Form 1**: split the node "+f'{node}? If "yes", which predecessor and successor?')
                
            st.radio("split the node?", ("yes", "no"), disabled=True)
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("predecessor", [], 
                                                      key = "predecessor_select_box",
                                                      disabled=True)
            with col2:
                st.selectbox("successor", [], 
                                                     key = "successor_select_box",
                                                     disabled=True)
                    
            st.form_submit_button("submit", disabled=True)
            
        add_and_delete_form_placeholder = st.empty()           
        add_and_delete_form = add_and_delete_form_placeholder.form(f"add and delete form")
        with add_and_delete_form:
            node = session_state["node"]
            st.write(f"**Form 2**: add and delete edges for node {node}")               
            st.multiselect("edges to add", [], disabled=True)
            st.multiselect("edges to delete", [], disabled=True)
                            
            st.form_submit_button("submit", disabled=True)
    
    if not stop_and_clear_button and not session_state["stop_and_clear"]:
        graph_manipulation_process(session_state, LOG_FILE_PATH, LOG_FILE_PATH2, HTML_IMG_PATH, GRAPH_MANIPULATION_SEED, 
                               split_the_node_form_placeholder, add_and_delete_form_placeholder)
    else:
        print_INFO_message_timestamp("Stop and clear state")
        return

def graph_manipulation(session_state, TIMES):
    placeholder_button = st.container()
    st.markdown("---")
    placeholder_error = st.empty()
    placeholder = st.empty()

    with placeholder_button:
        col1, col2, col3, col4 = st.columns(4)
    
        with col1:
            button_load = st.button("Load data for graph manipulation")
        with col2:
            button_manipulation = st.button("Start graph manipulation process")
        with col3:
            button_refine = st.button("Refine modified graph", on_click=on_submit_refine, args=(placeholder,))
        with col4:
            button_apply = st.button("Apply modification to all graphs", on_click=on_submit_apply, args=(placeholder,))

    ############################################## ERROR SECTION ##############################################
    with placeholder_error:
        if st.session_state["load_data_error"]:
            st.error("Please load data first!", icon="🚨")
        else:
            st.write("")
    
    ############################################## LOAD DATA ##############################################
    if button_load:
        placeholder_error.empty()
        graph_manipulation_load_data(session_state, TIMES)
        initialize_session_state_attributes(True)
        clear_log_files()
        session_state["button_load"] = True
        
        if button_manipulation:
            st.session_state["stop_and_clear"] = True
    
    ############################################## MODIFY GRAPH ##############################################    
    if button_manipulation and not st.session_state["load_data_error"]:
        for att in ["average_graphs", "node", "node_mapping", "predecessors_id", "successors_id", "stop_and_clear", "button_load"]:
            if att not in st.session_state:
                return st.error("Please load data first!", icon="🚨")
        
        with placeholder:
            graph_manipulation_process_template(session_state, TIMES, 
                                   LOG_FILE_PATH, LOG_FILE_PATH2, HTML_IMG_PATH, GRAPH_MANIPULATION_SEED)
            session_state["button_load"] = False
            session_state["stop_and_clear"] = False

        if session_state["stop_and_clear"] or stop_and_clear_button:
            placeholder.warning("Process interrupted and state cleared (load the data to start again)", icon="❌")
        else:
            placeholder.success("Process completed: changes has been saved. Download data using the download button", icon="✅")
        # if session_state["button_load"]:
        #     placeholder.warning("Process interrupted (load the data to start again)", icon="❌")
        # elif session_state["stop_and_clear"]:
        #     placeholder.warning("Process interrupted and state cleared (load the data to start again)", icon="❌")
        # else:
        #     placeholder.success("Process completed: changes has been saved. Download data using the download button", icon="✅")

    ############################################## REFINE GRAPH ############################################## 
    if "is_submitted" in session_state["refine_graph"].keys() and not st.session_state["load_data_error"]:
        if session_state["refine_graph"]["is_submitted"]:
            placeholder = st.container()
            with placeholder:
                graph_col, _, form_col = st.columns([2,0.25,1])
                                
                with graph_col:
                    st.plotly_chart(session_state["refine_graph"]["fig"], use_container_width=True)
                        
                with form_col:
                    for i in range(5):
                        st.write("#")
                    refine_form_placeholder = st.empty()           
                    G = session_state["refine_graph"]["G"]
                    refine_graph(G, refine_form_placeholder, session_state)

    ############################################## APPLY GRAPH CHANGES ##############################################
    if st.session_state["apply_graph_modification"] and not st.session_state["load_data_error"]:
        with placeholder:
            st.subheader("Load history changes data")
            col1, _, col2, _ = st.columns([1,0.25,1, 0.25])
            with col1:
                manipulation_data = st.file_uploader("**Upload history changes (from manipulation section)**", 
                                             type=["pkl", "bin"], 
                                             key="upload_button_manipulation",)
                
                if st.session_state["upload_button_manipulation"] is not None:
                    st.session_state["history_changes"] = pkl.load(st.session_state["upload_button_manipulation"])
            with col2:
                refine_data = st.file_uploader("**Upload history changes (from refine section)**", 
                                             type=["pkl", "bin"], 
                                             key="upload_button_refine",)
                if st.session_state["upload_button_refine"] is not None:
                    st.session_state["history_changes_refine"] = pkl.load(st.session_state["upload_button_refine"])
        
        for i in range(3):
            st.write("#")
        st.button("Apply changes", on_click=on_submit_apply2)    
        
    if "is_submitted2" in session_state["refine_graph"].keys() and not st.session_state["load_data_error"]:
        if session_state["refine_graph"]["is_submitted2"]:
            print("Apply changes")
            progress_bar_sub_placeholder = st.empty()
            graph_manipulation_process(st.session_state, 
                                        LOG_FILE_PATH, LOG_FILE_PATH2, HTML_IMG_PATH, GRAPH_MANIPULATION_SEED,
                                        st.empty(), st.empty(), apply_to_all=True, progress_bar_placeholder=progress_bar_sub_placeholder)
            refine_graph(session_state["modified_graph"], st.empty(), session_state, apply_to_all=True)
        
            
            
        
            
        
        
# -------------------------------------------- DETEMINISTIC ANALYSIS --------------------------------------------
def deterministic_load_data(session_state, TIMES, facilities_number):
    c = 0
    
    progress_bar = st.progress(0, "Loading data...")
    if f"fls_exact_{facilities_number}" not in session_state:
        fls_exact = {}
        for i, time in enumerate(TIMES):
            print_INFO_message_timestamp(f"Loading deterministic solution for: {time}")
            progress_bar.progress((i+1)*1/len(TIMES), f"Loading exact solution for: {time}")
            path = project_path+r"/"+retrieve_light_solution_path(facilities_number, time)
            fls_exact[time] = FacilityLocation.load(path)
            
        session_state[f"fls_exact_{facilities_number}"] = fls_exact
    c += 1
        
    if f"dfs_{facilities_number}" not in session_state:
        root = project_path+rf"/data/08_reporting/{facilities_number}_locations"
        paths = [p for p in os.listdir(root) if ("solution_vs_scenario" in p) and ("worst" not in p)]
            
        dfs = {}

        for path in paths:
            with open(os.path.join(root, path), "rb") as f:
                key = tuple(path.
                        replace("all_day_free_flow", "all-day-free-flow").
                        replace("all_day", "all-day").
                        removesuffix(".pkl").split("_")[-3:])
                        
                dfs[key] = pkl.load(f)
            
        session_state[f"dfs_{facilities_number}"] = dfs
    c += 1
            
    # if f"dfs_worst_{facilities_number}" not in session_state:
    #     root = project_path+rf"/data/08_reporting/{facilities_number}_locations"
    #     paths_worst = [p for p in os.listdir(root) if ("solution_vs_scenario" in p) and ("worst" in p)]
            
    #     dfs_worst = {}

    #     for path in paths_worst:
    #         with open(os.path.join(root, path), "rb") as f:
    #             key =   tuple(path.
    #                     replace("all_day_free_flow", "all-day-free-flow").
    #                     replace("all_day", "all-day").
    #                     removesuffix(".pkl").split("_")[-4:-1])
                        
    #             dfs_worst[key] = pkl.load(f)
            
    #     session_state[f"dfs_worst_{facilities_number}"] = dfs_worst
    c+=1

    if f"average_graphs" not in session_state:
        
        average_graphs = {}
        data_for_traffic_jam_visualization = {}
        for time in TIMES[1:]:
            path = project_path+"/"+retrieve_average_graph_path(time, True, True, True, False)
            with open(path, "rb") as f:
                average_graphs[time] = pkl.load(f)
            
            data_for_traffic_jam_visualization[time] = prepare_data_for_traffic_jam_visualization(average_graphs[time])

        session_state[f"average_graphs"] = average_graphs
        session_state[f"data_for_traffic_jam_visualization"] = data_for_traffic_jam_visualization
        
    c+=1

    if c == 4:
        progress_bar.progress(100, "Loading data completed!")

def deterministic_generate_viz(session_state, TIMES, facilities_number):
    if f"fls_exact_{facilities_number}" not in session_state:
        return st.error("Please load data first!", icon="🚨")

    # --------------------------------- TRAFFIC JAM ------------------------------------------
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    average_graphs = session_state[f"average_graphs"]
    data_for_traffic_jam_visualization = session_state[f"data_for_traffic_jam_visualization"]
    jam_vector = []
    for _, _, diff_weight in data_for_traffic_jam_visualization.values():
        jam_vector.append(diff_weight)
    
    for key, col in zip(TIMES[1:], [col1, col2, col3, col4]):
        if f"map_traffic_jam_{key}" not in session_state:
            map = show_traffic_jam(average_graphs[key], 
                                   display_jam=True, 
                                   title="TRAFFIC JAM - " + key,
                                   precomputed_data=data_for_traffic_jam_visualization[key],
                                   overall_jam = np.mean(jam_vector, axis=0),
                                   )
            session_state[f"map_traffic_jam_{key}"] = map
        with col:
            st.plotly_chart(
                session_state[f"map_traffic_jam_{key}"],
                use_container_width=True
            )
    
    
    #------------------------------- FACILITIES ON MAP ---------------------------------------
    col1, col2 = st.columns([1,1])
    dfs = session_state[f"dfs_{facilities_number}"]
    # dfs_worst = session_state[f"dfs_worst_{facilities_number}"]
    session_state[f"df_min_{facilities_number}"] = compute_min_distance_df(dfs, None)#, dfs_worst)

    if f"facilities_on_map_{facilities_number}" not in session_state:
        fls_exact = session_state[f"fls_exact_{facilities_number}"]
        fig = facilities_on_map([fl for fl in fls_exact.values()], 
                                    extra_text=[time for time in fls_exact.keys()],
                                    title_pad_l=200)
        session_state[f"facilities_on_map_{facilities_number}"] = fig
    
    if f"map_longest_paths_{facilities_number}" not in session_state:
        dfs = session_state[f"dfs_{facilities_number}"]
        average_graphs = session_state[f"average_graphs"]
        map = visualize_longest_paths(dfs, average_graphs)
        session_state[f"map_longest_paths_{facilities_number}"] = map
        
    with col1:
        st.plotly_chart(session_state[f"facilities_on_map_{facilities_number}"], 
                        use_container_width=True)

    #---------------------------------- MAP LONGEST PATH -------------------------------------        

    with col2:
        st.markdown("<h3 style='text-align: center;'>Longest paths</h3>", unsafe_allow_html=True)
        st_folium(
                session_state[f"map_longest_paths_{facilities_number}"],
                returned_objects=[],
                width=800)

    #------------------ FREE FLOW SOLUTION UNDER DIFFERENT SCENARIOS COMPARISON ------------------
    #------------------ OBJ FUNCTION VALUE -------------
    col1, col2 = st.columns(2)
    if f"abs_diff_barplot_{facilities_number}" not in session_state:
        fls_exact = session_state[f"fls_exact_{facilities_number}"]
        dfs = session_state[f"dfs_{facilities_number}"]
        try:
            dfs_worst = session_state[f"dfs_worst_{facilities_number}"]
        except:
            dfs_worst = None
            
        a = list(range(len(TIMES)-1))
        b = list(range(len(TIMES)-1))
        b_worst = list(range(len(TIMES)-1))

        for i, time in enumerate(TIMES[1:]):
            a[i], b[i], b_worst[i] = compute_rel_diff(fls_exact, dfs, dfs_worst, time)
        session_state[f"abs_diff_barplot_{facilities_number}"] = (a,b,b_worst)

    with col1:
        (a,b,b_worst) = session_state[f"abs_diff_barplot_{facilities_number}"]
        fig = objective_function_value_under_different_cases(a, b, b_worst)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        with open(project_path+
                  rf"/data/09_streamlit_md/Deterministic_results/{facilities_number} facilities/sideBySideWithFirstBarplot.md", 
                  "r",
                  encoding="utf-8") as f:
            content = f.read()

        for i in range(6):
            st.write("")
        st.markdown(content)
    
    #------------------ RELATIVE DIFFERENCES ------------  
    col1, col2 = st.columns(2)
    with col1:
        with open(project_path+rf"/data/09_streamlit_md/Deterministic_results/{facilities_number} facilities/sideBySideWithSecondBarplot.md", 
                  "r",
                  encoding="utf-8") as f:
            content = f.read()

        for i in range(6):
            st.write("")
        st.markdown(content)
    
    with col2:    
        (a,b,b_worst) = session_state[f"abs_diff_barplot_{facilities_number}"]
        fig = outsample_evaluation_relative_differences(a, b, b_worst)
        st.plotly_chart(fig, use_container_width=True)

    #------------------------------------ DISTRIBUTION ANALYSIS ---------------------------------------
    col1, col2 = st.columns(2)
    if f"distribution_violin_plot_{facilities_number}" not in session_state:
        df_min = session_state[f"df_min_{facilities_number}"]
        fig = average_travel_time_across_under_different_cases(df_min)
        session_state[f"distribution_violin_plot_{facilities_number}"] = fig

    with col1:
        st.plotly_chart(session_state[f"distribution_violin_plot_{facilities_number}"], 
        use_container_width=True)
        
        
        
    df_min = session_state[f"df_min_{facilities_number}"]
            
    fig = travel_times_distribution_under_different_cases(df_min)
    st.plotly_chart(fig, use_container_width=True)
        
def deterministic_analysis(session_state, TIMES, facilities_number, ratio1, ratio2, seed):
    ############################################## RUN THE MODEL ##############################################
    # button1 = st.button("Run the model")
        
    # if button1:
    #     with open(r"/app/geospatial-analysis/facility-location-Bergen/conf/base/parameters/fl_deterministic.yml", "w") as f:
    #         yaml.dump({
    #             "fl_deterministic.data":{
    #             "facilities_number": facilities_number,
    #             "ratio1": ratio1,
    #             "ratio2": ratio2,
    #             "seed": seed
    #             }
    #         }, f)
                
    #     n_facilities = facilities_number
            
    #     for time in TIMES:
    #         need_to_run = False
    #         path = r"/app/geospatial-analysis/facility-location-Bergen/"+retrieve_light_solution_path(n_facilities, time)
    #         if os.path.exists(path) == False:
    #             need_to_run = True
    #             st.write(f"The model will run for {time} data")
    #         else:
    #             st.write(f"The model has already been run for {time} data")
            
    #     if need_to_run:
    #     # Create an instance of KedroSession
    #         with KedroSession.create(metadata.package_name) as session:
    #             # Load the Kedro project context
    #             context = session.load_context()
    #             pipelines = find_pipelines()
    #             runner = SequentialRunner( )
    #             otput_data = runner.run(pipelines["fl_deterministic"], catalog=context.catalog)
    #             message = otput_data["fl_deterministic.message"]
                    
    #         st.write(message+"!")
                
        
    # st.markdown("---")
        
    ############################################## RUN SOLUTION COMPARISON ##############################################
    # button2 = st.button("Process data for solution analysis")
        
    # TIME_SOLUTION = "all_day_free_flow"
        
    # if button2:
    #     for i, time in enumerate(TIMES):
    #         if time == "all_day_free_flow":
    #             time_scenario = "all_day"
    #             weight = "weight2"
    #         else:
    #             time_scenario = time
    #             weight = "weight"

    #         path = r"/app/geospatial-analysis/facility-location-Bergen/"+retrieve_solution_vs_scenario_path(facilities_number, TIME_SOLUTION, time_scenario, weight)
                
    #         if os.path.exists(path) == False:
    #             st.write(f"Start preprocessing for {time} solution data...")
    #             with open(r"/app/geospatial-analysis/facility-location-Bergen/conf/base/parameters/solution_comparison.yml", "w") as f:
    #                 yaml.dump({
    #                     f"solution_comparison.{2*i}":{
    #                     "time_solution": TIME_SOLUTION,
    #                     "time_scenario": time_scenario,
    #                     "facilities_number": facilities_number,
    #                     "weight": weight,
    #                     "worst": "False"
    #                     },
    #                 }, f)
                        
    #                 if weight != "weight2":
    #                     yaml.dump({
    #                     f"solution_comparison.{2*i+1}":{
    #                     "time_solution": TIME_SOLUTION,
    #                     "time_scenario": time_scenario,
    #                     "facilities_number": facilities_number,
    #                     "weight": weight,
    #                     "worst": "True"
    #                     }}, f)
                        
    #             # Create an instance of KedroSession
    #             with KedroSession.create(metadata.package_name) as session:
    #                 # Load the Kedro project context
    #                 context = session.load_context()
    #                 pipelines = find_pipelines()
    #                 runner = SequentialRunner( )
    #                 otput_data = runner.run(pipelines["solution_comparison"], catalog=context.catalog)
                    
    #             st.write("Done!")
                    
    #         else:
    #             st.write(f"Preprocessing for {time} solution data has already been done")
                
    #     st.write("Preprocessing for all the scenarios has been completed!")
            
    # st.markdown("---")
    
    col1, col2, _, _ = st.columns(4)
    with col1:
        button_load = st.button("Load data for solution analysis")
    with col2:
        button_viz = st.button("Generate vizualizations")
    st.markdown("---")

    ############################################## LOAD DATA ##############################################
    if button_load:
        deterministic_load_data(session_state, TIMES, facilities_number)
        
    ############################################## GENERATE VIZ ##############################################    
    if button_viz:
        deterministic_generate_viz(session_state, TIMES, facilities_number)

# -------------------------------------------- STOCHASTIC ANALYSIS ---------------------------------------------
def stochastic_load_data(session_state, facilities_number):
    root_path = project_path+r"/data/07_model_output"
    
    if f"fls_stochastic_{facilities_number}" not in session_state:
        fls_solutions = {}
        fls_solutions["stochastic"] = StochasticFacilityLocation.load(root_path+f"/{facilities_number}_locations/stochastic_solution/lshape_solution.pkl")
        fls_solutions["deterministic"] = FacilityLocation.load(root_path+f"/{facilities_number}_locations/deterministic_exact_solutions/super_light_exact_solution_all_day_free_flow.pkl")
        session_state[f"fls_stochastic_{facilities_number}"] = fls_solutions  

def stochastic_load_metrics(session_state):
    root_path = project_path+r"/data/07_model_output"
    if f"df_metrics" not in session_state:
        df_metrics = pd.read_csv(root_path+f"/stochastic_solution_evaluation_metrics.csv")
        new_cols_name = ["n_locations"]
        for col in df_metrics.columns[1:]:
            new_cols_name.append(col+" (min)")    
            try:
                df_metrics[col] = df_metrics[col]/60
                df_metrics[col] = df_metrics[col].round(2)
            except:
                continue
        
        df_metrics.columns = new_cols_name
        
        session_state[f"df_metrics"] = df_metrics 

def stochastic_generate_viz(session_state, facilities_number):
    if f"fls_stochastic_{facilities_number}" not in session_state:
        st.error("Please load data first!", icon="🚨")
        return go.Figure()

    fl_stochastic = session_state[f"fls_stochastic_{facilities_number}"]["stochastic"]
    fl_deterministic = session_state[f"fls_stochastic_{facilities_number}"]["deterministic"]
    fls = [fl_stochastic, fl_deterministic]
    
    return facilities_on_map(fls)

def stochastic_analysis(session_state):
    col1, col2, col3, _ = st.columns(4)
    with col1:
        button_load = st.button("Load data for solution analysis")
    
    with col2:
        button_viz = st.button("Generate vizualizations")
    with col3:
        button_metrics = st.button("Generate metrics")
    st.markdown("---")
        
    ############################################## LOAD DATA ##############################################
    if button_load:
        progress_bar = st.progress(1/3, "Loading stochastic solution...")
        for i, facilities_number in enumerate(FACILITIES_NUMBER):
                stochastic_load_data(session_state, facilities_number)
        progress_bar.progress(2/3, "Loading stochastic solutions metrics...")
        stochastic_load_metrics(session_state)
        progress_bar.progress(3/3, "Loading data completed!")
    
    ############################################## GENERATE VIZ ##############################################    
    if button_viz:
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
    
        for facilities_number in FACILITIES_NUMBER:
            fig = stochastic_generate_viz(session_state, facilities_number)
            with cols[facilities_number-1]:
                st.plotly_chart(fig, use_container_width=True)

    ############################################## GENERATE METRICS ##############################################
    if button_metrics:
        col1, col2, col3 = st.columns([1,2,1])
        cols = [col1, col2, col3]
        if session_state.get(f"df_metrics") is None:
            with col1:  
                st.error("Please load data first!", icon="🚨")
        else:
            with col2:
                df_metrics = session_state[f"df_metrics"]
                st.dataframe(df_metrics)

@st.cache_data
def read_project_description(project_path):
    with open(project_path+r"/data/09_streamlit_md/Project description.md", "r") as f:
        content = f.read()
    return content

@st.cache_data
def read_theoretical_framework(project_path):
    with open(project_path+r"/data/09_streamlit_md/Theoretical framework.md", "r", encoding="utf-8") as f:
        content = f.read()
    return content

if __name__ == '__main__':
    initialize_session_state_attributes()
    side_bar = st.sidebar

    with side_bar:
        st.title("Control Panel")

        st.subheader("Section")
        
        section = st.selectbox(
                "Section selection",
                ("Project description", 
                 "Theoretical Framework", 
                 "Graph manipulation",
                 "Deterministic models analysis", 
                 "Stochastic models analysis"),
                label_visibility="collapsed",)
        
        
        if section == "Deterministic models analysis" or section == "Stochastic models analysis":
            st.subheader("Parameters for the optimization model")
            if section == "Deterministic models analysis":
                st.markdown("**Facilities number:**")
                facilities_number = st.radio(
                    "Facilities number",
                    (1, 2, 3),
                    horizontal=True,
                    label_visibility="collapsed",)
            else:
                st.markdown("**Facilities number:**")
                facilities_number = st.radio(
                    "Facilities number",
                    ([1,2,3],),
                    horizontal=True,
                    label_visibility="collapsed",)
            
            st.markdown("**Ratio for customers locations:**")
            ratio1 = st.radio(
                "Ratio for customers locations",
                (1/5,),
                label_visibility="collapsed")

            st.markdown("**Ratio for candidate locations:**")
            ratio2 = st.radio(
                "Ratio for candidate locations",
                (1/10,),
                label_visibility="collapsed")
                
            st.markdown("**Seed for reproducibility:**")
            seed = st.radio(
                    "Seed for reproducibility",
                    (324324,),
                    label_visibility="collapsed",)
            
            
        if section == "Graph manipulation":
            st.subheader("Restore the old state")
            uploaded_file = st.file_uploader("**Upload graph**", 
                                             type=["pkl", "bin"], 
                                             key="upload_button_0",)
            uploaded_file = st.file_uploader("**Upload history changes**", 
                                             type=["pkl", "bin"], 
                                             key="upload_button_1",)
            
            st.subheader("Save the current state")
            st.download_button("download modified graph",
                               pkl.dumps(session_state["modified_graph"]),
                              file_name="graph.pkl")
            st.download_button("download history changes",
                               pkl.dumps(session_state["history_changes"]),
                              file_name="history_changes.pkl")

            
    if section not in ["Project description", "Theoretical Framework"]:
        st.title("Facility Location dashboard")
        st.markdown("---")

    if section == "Project description":
        content = read_project_description(project_path)
        col1, col2, col3 = st.columns([1,2.5,1])
        with col2:
            st.markdown(content)
            
    elif section == "Theoretical Framework":
        content = read_theoretical_framework(project_path)
        col1, col2, col3 = st.columns([1,2.5,1])
        with col2:
            st.markdown(content)
            
    elif section == "Graph manipulation":
        graph_manipulation(session_state, TIMES)
        
    elif section == "Deterministic models analysis":
        deterministic_analysis(session_state, TIMES, facilities_number, ratio1, ratio2, seed)
        
    elif section == "Stochastic models analysis":
        stochastic_analysis(session_state)
