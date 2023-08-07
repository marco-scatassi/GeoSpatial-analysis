import sys

# Get the directory path to add to PYTHONPATH
directory_path = r"\\Pund\Stab$\guest801951\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
if directory_path not in sys.path:
    sys.path.append(directory_path)
    
import os
import pickle as pkl
import random
import time as ptime
from copy import deepcopy
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from facility_location import (
    FacilityLocation,
    FacilityLocationReport,
    StochasticFacilityLocation,
)
from graph_manipulation import *

# from kedro.framework.project import find_pipelines
# from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
# from kedro.pipeline import Pipeline, pipeline
# from kedro.runner import SequentialRunner
from log import print_INFO_message, print_INFO_message_timestamp
from retrieve_global_parameters import (
    retrieve_average_graph_path,
    retrieve_light_solution_path,
    retrieve_solution_vs_scenario_path,
)
from src.facility_location_Bergen.custome_modules.graphical_analysis import (
    average_travel_time_across_under_different_cases,
    compute_min_distance_df,
    compute_rel_diff,
    facilities_on_map,
    objective_function_value_under_different_cases,
    outsample_evaluation_relative_differences,
    travel_times_distribution_under_different_cases,
    visualize_longest_paths,
)
from streamlit_folium import st_folium

st.set_page_config(layout = "wide")
session_state = st.session_state

project_path = r"\/Pund/Stab$/guest801951/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen"
metadata = bootstrap_project(project_path)

TIMES = ["all_day_free_flow", "all_day", "morning", "midday", "afternoon"]
FACILITIES_NUMBER = [1,2,3]

LOG_FILE_PATH = r"\\Pund\Stab$\guest801951\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\logs\split_roads.log"
LOG_FILE_PATH2 = r"\\Pund\Stab$\guest801951\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\logs\split_roads_changes.log"
HTML_IMG_PATH = r"\\Pund\Stab$\guest801951\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\logs\img_split_roads.html"

GRAPH_MANIPULATION_SEED=8797

# --------------------------------------------- UTILITY AND CALLBACK --------------------------------------------
def initialize_session_state_attributes():
    st.session_state["node"] = "___"
    st.session_state["node_mapping"] = {}
    st.session_state["predecessors_id"] = []
    st.session_state["successors_id"] = []
    st.session_state["is_split_the_node_form_submitted"] = False
    st.session_state["stop_and_save"] = False
    st.session_state["button_load"] = False
 
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
                <h1>Loacal graph visualization</title>
                <p>Searching for the first node to analyze...</p>
                <p>Update widgets when ready!</p>
                </body>
                </html>""")
    
def stop_and_save_callback():
    st.session_state["stop_and_save"] = True
    initialize_session_state_attributes()
    clear_log_files()


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
                               split_the_node_form_placeholder, add_and_delete_form_placeholder, update_widgets_placeholder):
    
    F = deepcopy(session_state[f"average_graphs"]["all_day"])
    history_changes = session_state["history_changes"]
    
    nodes = list(F.nodes())
    seed = random.seed(GRAPH_MANIPULATION_SEED)

    # if history_changes == {}:
    #     origin = random.choice(nodes)
    # else:
    #     origin = list(history_changes.keys())[-1]
    
    origin = random.choice(nodes)
    
    if not session_state["stop_and_save"]:
        print_INFO_message_timestamp("Splitting two way roads")
        split_two_way_roads(F, 
                            origin=origin, 
                            session_state=session_state,
                            split_the_node_form_placeholder=split_the_node_form_placeholder,
                            add_and_delete_form_placeholder=add_and_delete_form_placeholder,
                            update_widgets_placeholder=update_widgets_placeholder,
                            count=0,
                            count_max=100, 
                            log_file_path=LOG_FILE_PATH,
                            log_file_path2=LOG_FILE_PATH2, 
                            img_path=HTML_IMG_PATH,)
    else:
        print_INFO_message_timestamp("Stop and save changes")
        session_state["stop_and_save"] = False
        return
    
def graph_manipulation_process_template(session_state, TIMES, 
                               LOG_FILE_PATH, LOG_FILE_PATH2, HTML_IMG_PATH, GRAPH_MANIPULATION_SEED):
    
    text_col, img_col = st.columns(2)
    with open(HTML_IMG_PATH, "r", encoding="utf-8") as f:
        html_img = f.read()
            
    with img_col:
        st.components.v1.html(html_img, height=600)
        _, button_col, _ = st.columns(3)
        with button_col:
            stop_and_save_button = st.button("Stop and save changes", on_click=stop_and_save_callback)
            
    with text_col:
        split_the_node_form_placeholder = st.empty()
        split_the_node_form = split_the_node_form_placeholder.form(f"split the node form")
        with split_the_node_form:
            node = session_state["node"]
                
            st.write("split the node "+f'{node}? If "yes", which predecessor and successor?')
                
            split_the_node = st.radio("split the node?", ("yes", "no"), disabled=True)
            col1, col2 = st.columns(2)
            with col1:
                selcted_predecessor_id = st.selectbox("predecessor", [], 
                                                      key = "predecessor_select_box",
                                                      disabled=True)
            with col2:
                selected_successor_id = st.selectbox("successor", [], 
                                                     key = "successor_select_box",
                                                     disabled=True)
                    
            submit = st.form_submit_button("submit", disabled=True)
            
        add_and_delete_form_placeholder = st.empty()           
        add_and_delete_form = add_and_delete_form_placeholder.form(f"add and delete form")
        with add_and_delete_form:
            node_mapping = session_state['node_mapping']
            node_mapping_r = {v: k for k, v in node_mapping.items()}
            node = session_state["node"]
            history_changes = session_state["history_changes"]
                
            st.write(f"add and delete edges for node {node}")
            edge_list = [None]
            for node1 in node_mapping.keys():
                for node2 in node_mapping.keys():
                    if session_state["modified_graph"].has_edge(node1, node2):
                        edge_list.append((node_mapping[node1], node_mapping[node2]))
                            
            edges_to_add = st.multiselect("edges to add", edge_list, disabled=True)
            edges_to_delete = st.multiselect("edges to delete", edge_list, disabled=True)
                            
            submit = st.form_submit_button("submit", disabled=True)
            if submit:
                if None in edges_to_add:
                    if len(edges_to_add) > 1:
                        edges_to_add.remove(None)
                        history_changes[node]["new_edges"] = [(node_mapping_r[e[0]], node_mapping_r[e[1]]) for e in edges_to_add]
                    else:
                        history_changes[node]["new_edges"] = []
                if None in edges_to_delete:
                    if len(edges_to_delete) > 1:
                        edges_to_delete.remove(None)
                        history_changes[node]["edges_to_delete"] = [(node_mapping_r[e[0]], node_mapping_r[e[1]]) for e in edges_to_delete]
                    else:
                        history_changes[node]["edges_to_delete"] = []
                                
                print_INFO_message_timestamp(f'new edges: {history_changes[node]["new_edges"]}', LOG_FILE_PATH)
                print_INFO_message_timestamp(f'edges to delete: {history_changes[node]["edges_to_delete"]}', LOG_FILE_PATH)

        
        update_widgets_placeholder = st.empty()
        update_widgets_placeholder.write("Searching for the next node to analyze...")
                
        # session_state_container = st.empty()
        # session_state_container.json(session_state)
        
    graph_manipulation_process(session_state, LOG_FILE_PATH, LOG_FILE_PATH2, HTML_IMG_PATH, GRAPH_MANIPULATION_SEED, 
                               split_the_node_form_placeholder, add_and_delete_form_placeholder, update_widgets_placeholder)

def graph_manipulation(session_state, TIMES):
    col1, col2, _, _ = st.columns(4)
    placeholder = st.empty()
    
    with col1:
        button_load = st.button("Load data for graph manipulation")
    with col2:
        button_manipulation = st.button("Start graph manipulation process")
    st.markdown("---")
    
    ############################################## LOAD DATA ##############################################
    if button_load:
        graph_manipulation_load_data(session_state, TIMES)
        initialize_session_state_attributes()
        clear_log_files()
        session_state["button_load"] = True
        
        if button_manipulation:
            st.session_state["stop_and_save"] = True
    
    ############################################## GENERATE VIZ ##############################################    
    if button_manipulation:
        for att in ["node", "node_mapping", "predecessors_id", "successors_id", "is_split_the_node_form_submitted", "stop_and_save", "button_load"]:
            if att not in st.session_state:
                return st.error("Please load data first!", icon="🚨")
            
        with placeholder:
            graph_manipulation_process_template(session_state, TIMES, 
                                   LOG_FILE_PATH, LOG_FILE_PATH2, HTML_IMG_PATH, GRAPH_MANIPULATION_SEED)
        
        if session_state["button_load"]:
            placeholder.warning("Process interrupted", icon="❌")
        else:
            placeholder.success("Process completed: changes has been saved", icon="✅")
        
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
            
    if f"dfs_worst_{facilities_number}" not in session_state:
        root = project_path+rf"/data/08_reporting/{facilities_number}_locations"
        paths_worst = [p for p in os.listdir(root) if ("solution_vs_scenario" in p) and ("worst" in p)]
            
        dfs_worst = {}

        for path in paths_worst:
            with open(os.path.join(root, path), "rb") as f:
                key =   tuple(path.
                        replace("all_day_free_flow", "all-day-free-flow").
                        replace("all_day", "all-day").
                        removesuffix(".pkl").split("_")[-4:-1])
                        
                dfs_worst[key] = pkl.load(f)
            
        session_state[f"dfs_worst_{facilities_number}"] = dfs_worst
    c+=1

    if f"average_graphs" not in session_state:
        
        average_graphs = {}
        for time in TIMES[1:]:
            path = project_path+"/"+retrieve_average_graph_path(time, connected=True)
            with open(path, "rb") as f:
                average_graphs[time] = pkl.load(f)

        session_state[f"average_graphs"] = average_graphs

    c+=1

    if c == 4:
        progress_bar.progress(100, "Loading data completed!")

def deterministic_generate_viz(session_state, TIMES, facilities_number):
    if f"fls_exact_{facilities_number}" not in session_state:
        return st.error("Please load data first!", icon="🚨")

    #------------------------------- FACILITIES ON MAP ---------------------------------------
    col1, col2 = st.columns([1.5,1])
    dfs = session_state[f"dfs_{facilities_number}"]
    dfs_worst = session_state[f"dfs_worst_{facilities_number}"]
    session_state[f"df_min_{facilities_number}"] = compute_min_distance_df(dfs, dfs_worst)

    if f"facilities_on_map_{facilities_number}" not in session_state:
        fls_exact = session_state[f"fls_exact_{facilities_number}"]
        fig = facilities_on_map([fl for fl in fls_exact.values()], 
                                    extra_text=[time for time in fls_exact.keys()],
                                    title_pad_l=200)
        session_state[f"facilities_on_map_{facilities_number}"] = fig
        
    with col1:
        st.plotly_chart(session_state[f"facilities_on_map_{facilities_number}"], 
                        use_container_width=True)

    with open(project_path+rf"/data/09_streamlit_md/Deterministic_results/{facilities_number} facilities/sideBysideWithMap.md", "r") as f:
        content = f.read()

    with col2:
        for i in range(7):
            st.write("")
        st.markdown(content)
        
    with open(project_path+rf"/data/09_streamlit_md/Deterministic_results/{facilities_number} facilities/underTheMap.md", "r") as f:
        content = f.read()

    st.markdown(content)

    #---------------------------------- MAP LONGEST PATH -------------------------------------
    col1, col2 = st.columns([1.5,1])
    if f"map_longest_paths_{facilities_number}" not in session_state:
        dfs = session_state[f"dfs_{facilities_number}"]
        average_graphs = session_state[f"average_graphs"]
        map = visualize_longest_paths(dfs, average_graphs)
        session_state[f"map_longest_paths_{facilities_number}"] = map
        
    with col1:
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
        dfs_worst = session_state[f"dfs_worst_{facilities_number}"]
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
        fls_solutions["deterministic"] = FacilityLocation.load(root_path+f"/{facilities_number}_locations/deterministic_exact_solutions/super_light_exact_solution_all_day.pkl")
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