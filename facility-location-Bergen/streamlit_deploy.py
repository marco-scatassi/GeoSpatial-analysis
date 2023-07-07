import os
import yaml
import pandas as pd
import time as ptime
import pickle as pkl
from PIL import Image
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
from kedro.runner import SequentialRunner
from kedro.pipeline import Pipeline, pipeline
from kedro.framework.session import KedroSession
from kedro.framework.project import find_pipelines
from kedro.framework.startup import bootstrap_project
from src.facility_location_Bergen.custome_modules.log import print_INFO_message_timestamp, print_INFO_message
from src.facility_location_Bergen.custome_modules.facility_location import (
    FacilityLocation, 
    FacilityLocationReport,
    StochasticFacilityLocation)

from src.facility_location_Bergen.custome_modules.retrieve_global_parameters import (
    retrieve_light_solution_path,
    retrieve_solution_vs_scenario_path,
)

from src.facility_location_Bergen.custome_modules.graphical_analysis import (
    compute_rel_diff,
    facilities_on_map,
    compute_min_distance_df,
    objective_function_value_under_different_cases,
    travel_times_distribution_under_different_cases,
    average_travel_time_across_under_different_cases
)

st.set_page_config(layout = "wide")
session_state = st.session_state

project_path = r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen"
metadata = bootstrap_project(project_path)

TIMES = ["all_day_free_flow", "all_day", "morning", "midday", "afternoon"]
FACILITIES_NUMBER = [1,2,3]

# -------------------------------------------- DETEMINISTIC ANALYSIS --------------------------------------------
def deterministic_load_data(session_state, TIMES, facilities_number):
    button3 = st.button("Load data for solution analysis")
    
    if button3:
        if f"fls_exact_{facilities_number}" not in session_state:
            fls_exact = {}
            root = r"C:/Users/Marco/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/"
            
            for time in TIMES:
                print_INFO_message_timestamp(f"Loading exact solution for {time}")
                st.write(f"Loading exact solution for {time}")
                path = root+retrieve_light_solution_path(facilities_number, time)
                fls_exact[time] = FacilityLocation.load(path)
            
            session_state[f"fls_exact_{facilities_number}"] = fls_exact
            st.write("Data has been loaded")
        
        if f"dfs_{facilities_number}" not in session_state:
            root = rf"C:/Users/Marco/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/data/08_reporting/{facilities_number}_locations"
            paths = [p for p in os.listdir(root) if ("solution_vs_scenario" in p) and ("worst" not in p)]
            
            dfs = {}

            for path in paths:
                with open(os.path.join(root, path), "rb") as f:
                    key = tuple(path.removesuffix(".pkl").split("_")[-2:])
                    if key[0] == "day":
                        key = tuple(["all_day", key[1]])
                        
                    dfs[key] = pkl.load(f)
            
            session_state[f"dfs_{facilities_number}"] = dfs
            
        if f"dfs_worst_{facilities_number}" not in session_state:
            root = rf"C:/Users/Marco/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen/data/08_reporting/{facilities_number}_locations"
            paths_worst = [p for p in os.listdir(root) if ("solution_vs_scenario" in p) and ("worst" in p)]
            
            dfs_worst = {}

            for path in paths_worst:
                with open(os.path.join(root, path), "rb") as f:
                    key = tuple(path.removesuffix(".pkl").split("_")[-3:-1])
                    if key[0] == "day":
                        key = tuple(["all_day", key[1]])
                        
                    dfs_worst[key] = pkl.load(f)
            
            session_state[f"dfs_worst_{facilities_number}"] = dfs_worst
            
        st.write("Data has been loaded")

    st.markdown("---")

def deterministic_generate_viz(session_state, TIMES, facilities_number):
    button4 = st.button("Generate vizualizations")
    
    if button4:
        col1, col2 = st.columns([1.8,1])
        
        dfs = session_state[f"dfs_{facilities_number}"]
        dfs_worst = session_state[f"dfs_worst_{facilities_number}"]
        session_state[f"df_min_{facilities_number}"] = compute_min_distance_df(dfs, dfs_worst)
        
        with col1:
            fls_exact = session_state[f"fls_exact_{facilities_number}"]
            fig = facilities_on_map([fl for fl in fls_exact.values()], 
                                    extra_text=[time for time in fls_exact.keys()],
                                    title_pad_l=300)
            st.plotly_chart(fig, use_container_width=True)
            
        
        col1, col2 = st.columns(2)
        with col1:
            fls_exact = session_state[f"fls_exact_{facilities_number}"]
            dfs = session_state[f"dfs_{facilities_number}"]
            dfs_worst = session_state[f"dfs_worst_{facilities_number}"]
            
            a = list(range(len(TIMES)-1))
            b = list(range(len(TIMES)-1))
            b_worst = list(range(len(TIMES)-1))
            for i, time in enumerate(TIMES[1:]):
                a[i], b[i], b_worst[i] = compute_rel_diff(fls_exact, dfs, dfs_worst, time)

            fig = objective_function_value_under_different_cases(a, b, b_worst)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df_min = session_state[f"df_min_{facilities_number}"]
            fig = average_travel_time_across_under_different_cases(df_min)
            st.plotly_chart(fig, use_container_width=True)
        
        
        
        df_min = session_state[f"df_min_{facilities_number}"]
            
        fig = travel_times_distribution_under_different_cases(df_min)
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")

def deterministic_analysis(session_state, TIMES, facilities_number, ratio1, ratio2, seed):
    ############################################## RUN THE MODEL ##############################################
    # button1 = st.button("Run the model")
    
    # if button1:
    #     with open(r".\conf\base\parameters\fl_deterministic.yml", "w") as f:
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
    #         path = retrieve_light_solution_path(n_facilities, time)
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

    #         path = retrieve_solution_vs_scenario_path(facilities_number, TIME_SOLUTION, time_scenario, weight)
            
    #         if os.path.exists(path) == False:
    #             st.write(f"Start preprocessing for {time} solution data...")
    #             with open(r".\conf\base\parameters\solution_comparison.yml", "w") as f:
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
        
    ############################################## LOAD DATA ##############################################
    deterministic_load_data(session_state, TIMES, facilities_number)
    
    ############################################## GENERATE VIZ ##############################################    
    deterministic_generate_viz(session_state, TIMES, facilities_number)

# -------------------------------------------- STOCHASTIC ANALYSIS ---------------------------------------------
def stochastic_load_data(session_state, facilities_number):
    root_path = r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\07_model_output"
    
    if f"fls_stochastic_{facilities_number}" not in session_state:
        fls_solutions = {}
        fls_solutions["stochastic"] = StochasticFacilityLocation.load(os.path.join(root_path, 
                                                                                    f"{facilities_number}_locations\stochastic_solution\lshape_solution.pkl"))
        fls_solutions["deterministic"] = FacilityLocation.load(os.path.join(root_path, 
                                                                             f"{facilities_number}_locations\deterministic_exact_solutions\light_exact_solution_all_day_free_flow.pkl"))
        session_state[f"fls_stochastic_{facilities_number}"] = fls_solutions    

def stochastic_load_metrics(session_state):
    root_path = r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\data\07_model_output"
    
    if f"df_metrics" not in session_state:
        df_metrics = pd.read_csv(os.path.join(root_path, 
                                                     f"stochastic_solution_evaluation_metrics.csv"))
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
    if session_state.get(f"fls_stochastic_{facilities_number}") is None:
        st.write("Please load the data first")
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
        
    ############################################## LOAD DATA ##############################################
    if button_load:
        for facilities_number in FACILITIES_NUMBER:
                st.write(f"Loading data for {facilities_number} facilities...")
                stochastic_load_data(session_state, facilities_number)
        st.write("Loading stochastic solutions metrics...")
        stochastic_load_metrics(session_state)
        st.write("Loading for all the scenarios has been completed!")
    st.markdown("---")
    
    ############################################## GENERATE VIZ ##############################################    
    if button_viz:
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
    
        for facilities_number in FACILITIES_NUMBER:
            fig = stochastic_generate_viz(session_state, facilities_number)
            with cols[facilities_number-1]:
                st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    ############################################## GENERATE METRICS ##############################################
    if button_metrics:
        col1, col2, col3 = st.columns([1,2,1])
        cols = [col1, col2, col3]
        if session_state.get(f"df_metrics") is None:
            with col1:  
                st.write("Please load the data first")
        else:
            with col2:
                df_metrics = session_state[f"df_metrics"]
                st.dataframe(df_metrics)
    st.markdown("---")
    
if __name__ == '__main__':
    st.title("Facility Location dashboard")
    
    st.subheader("Choose the analysis to perform")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        analysis = st.radio(
            "Analysis",
            ("Deterministic", "Stochastic"),
            horizontal=False,
            label_visibility="hidden",)
    
    st.subheader("Set the parameters for the optimization model")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if analysis == "Deterministic":
            st.markdown("**Facilities number:**")
            facilities_number = st.radio(
            "Facilities number",
            (1, 2, 3),
            horizontal=True,
            label_visibility="hidden",)
        else:
            st.markdown("**Facilities number:**")
            facilities_number = st.radio(
            "Facilities number",
            ([1,2,3],),
            horizontal=True,
            label_visibility="hidden",)
    with col2:
        st.markdown("**Ratio for customers locations:**")
        ratio1 = st.radio(
        "Ratio for customers locations",
        (1/5,),
        label_visibility="hidden")

    with col3:
        st.markdown("**Ratio for candidate locations:**")
        ratio2 = st.radio(
        "Ratio for candidate locations",
        (1/10,),
        label_visibility="hidden")
        
    with col4:
        st.markdown("**Seed for reproducibility:**")
        seed = st.radio(
            "Seed for reproducibility",
            (324324,),
            label_visibility="hidden",)
      
    st.markdown("---")
    
    if analysis == "Deterministic":
        deterministic_analysis(session_state, TIMES, facilities_number, ratio1, ratio2, seed)
    
    if analysis == "Stochastic":
        stochastic_analysis(session_state)