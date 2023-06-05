import sys

sys.path.append(
    r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
)


import os
import yaml
import time as ptime
import streamlit as st
from pathlib import Path
from kedro.runner import SequentialRunner
from kedro.pipeline import Pipeline, pipeline
from kedro.framework.session import KedroSession
from kedro.framework.project import find_pipelines
from kedro.framework.startup import bootstrap_project
from log import print_INFO_message_timestamp, print_INFO_message
from facility_location import FacilityLocation, FacilityLocationReport
from retrieve_global_parameters import (
    retrieve_solution_path,
    retrieve_solution_vs_scenario_path,
)

st.set_page_config(layout = "wide")

metadata = bootstrap_project(Path.cwd())

TIMES = ["all_day_free_flow", "all_day", "morning", "midday", "afternoon"]

if __name__ == '__main__':
    st.title("Facility Location dashboard")
    st.subheader("Set the parameters for the optimization model")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Facilities number:**")
        facilities_number = st.radio(
        "Facilities number",
        (1, 2, 3),
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
    
    ############################################## RUN THE MODEL ##############################################
    button1 = st.button("Run the model")
    
    if button1:
        with open(r".\conf\base\parameters\fl_deterministic.yml", "w") as f:
            yaml.dump({
                "fl_deterministic.data":{
                "facilities_number": facilities_number,
                "ratio1": ratio1,
                "ratio2": ratio2,
                "seed": seed
                }
            }, f)
            
        n_facilities = facilities_number
        
        for time in TIMES:
            need_to_run = False
            path = retrieve_solution_path(n_facilities, time)
            if os.path.exists(path) == False:
                need_to_run = True
                st.write(f"The model will run for {time} data")
            else:
                st.write(f"The model has already been run for {time} data")
        
        if need_to_run:
        # Create an instance of KedroSession
            with KedroSession.create(metadata.package_name) as session:
                # Load the Kedro project context
                context = session.load_context()
                pipelines = find_pipelines()
                runner = SequentialRunner( )
                otput_data = runner.run(pipelines["fl_deterministic"], catalog=context.catalog)
                message = otput_data["fl_deterministic.message"]
                
            st.write(message+"!")
            
      
    st.markdown("---")
    
    ############################################## RUN SOLUTION COMPARISON ##############################################
    button2 = st.button("Process data for solution analysis")
    
    TIME_SOLUTION = "all_day_free_flow"
    
    if button2:
        for i, time in enumerate(TIMES):
            if time == "all_day_free_flow":
                time_scenario = "all_day"
                weight = "weight2"
            else:
                time_scenario = time
                weight = "weight"

            path = retrieve_solution_vs_scenario_path(facilities_number, TIME_SOLUTION, time_scenario, weight)
            
            if os.path.exists(path) == False:
                st.write(f"Start preprocessing for {time} solution data...")
                with open(r".\conf\base\parameters\solution_comparison.yml", "w") as f:
                    yaml.dump({
                        f"solution_comparison.{2*i}":{
                        "time_solution": TIME_SOLUTION,
                        "time_scenario": time_scenario,
                        "facilities_number": facilities_number,
                        "weight": weight,
                        "worst": "False"
                        },
                    }, f)
                    
                    if weight != "weight2":
                        yaml.dump({
                        f"solution_comparison.{2*i+1}":{
                        "time_solution": TIME_SOLUTION,
                        "time_scenario": time_scenario,
                        "facilities_number": facilities_number,
                        "weight": weight,
                        "worst": "True"
                        }}, f)
                    
                # Create an instance of KedroSession
                with KedroSession.create(metadata.package_name) as session:
                    # Load the Kedro project context
                    context = session.load_context()
                    pipelines = find_pipelines()
                    runner = SequentialRunner( )
                    otput_data = runner.run(pipelines["solution_comparison"], catalog=context.catalog)
                
                st.write("Done!")
                
            else:
                st.write(f"Preprocessing for {time} solution data has already been done")
            
            st.write("Preprocessing for all the scenarios has been completed!")
        
    st.markdown("---")
        
    ############################################## GENERATE VISUALIZATION ##############################################
    button3 = st.button("Generate visualization")
    
    if button3:
        fls_exact = {}

        for time in TIMES:
            print_INFO_message(f"Loading exact solution for {time}")
            st.write(f"Loading exact solution for {time}")
            path = retrieve_solution_path(facilities_number, time)
            fls_exact[time] = FacilityLocation.load(path)
            ptime.sleep(5)
        
        report_exact = FacilityLocationReport(fls_exact)
        fig = report_exact.graphical_keys_solutions_comparison()
        temp_path = r"facility-location-Bergen\data\00_temp"
        
        st.pyplot(fig)