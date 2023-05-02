"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.7
"""

import sys
sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')

from .nodes import *
from retrieve_global_parameters import *
from kedro.pipeline import Pipeline, node, pipeline

def create_child_pipeline(key, value) -> list:
    return pipeline([
        node(
            func=verify_gdf_already_created,
            inputs=[f"params:{key}"],
            outputs="gdf_already_created",
            name="verify_gdf_already_created"
        ),
        node(
            func=from_json_to_gdf,
            inputs=["db_name", f"params:{key}", "gdf_already_created", "paths_gdf"],
            outputs="trigger",
            name="from_json_to_gdf"
        ),
        node(
            func=update_data_catalog_gdf,
            inputs=[f"params:{key}", "trigger", "paths_gdf"],
            outputs="trigger_gdf",
            name="update_data_catalog_gdf"
        )
    ],
    namespace=f"convert_to_gdf.{value}",
    parameters={key:key})


def create_pipeline(**kwargs) -> Pipeline:
    conf_params = retrieve_global_parameters()
    
    child_pipelines = []
    
    for key, value in conf_params.items():
        if "convert_to_gdf.date" in key and conf_params[key] is not None :
            child_pipelines.append(create_child_pipeline(key, value))
    
    convert_to_gdf_pipeline = sum(child_pipelines)
    
    # --------- chain retrieve_global_params and visualization pipelines ---------
    mapping={}
    for e in convert_to_gdf_pipeline.all_inputs():
        if "db_name" in e:
            mapping[e]="db_name"
            
    return pipeline(pipe=convert_to_gdf_pipeline, inputs=mapping, tags=["convert_to_gdf"])