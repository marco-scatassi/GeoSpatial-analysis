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
            func=define_saving_paths_gdf,
            inputs=[f"params:{key}"],
            outputs="paths_gdf",
            name="define_saving_paths_gdf"
        ),
        node(
            func=verify_gdf_already_created,
            inputs=["paths_gdf"],
            outputs="gdf_already_created",
            name="verify_gdf_already_created"
        ),
        node(
            func=from_json_to_gdf,
            inputs=["db_name", f"params:{key}", "gdf_already_created", "paths_gdf"],
            outputs="trigger_gdf",
            name="from_json_to_gdf"
        ),
        node(
            func=update_data_catalog_gdf,
            inputs=[f"params:{key}", "trigger_gdf", "paths_gdf"],
            outputs=None,
            name="update_data_catalog_gdf"
        ),
        node(
          func=define_saving_paths_gif,
          inputs=[f"params:{key}"],  
          outputs="paths_gif",
          name="define_saving_paths_gif"
        ),
        node(
            func=create_and_save_animation,
            inputs=[f"params:{key}", "paths_gdf", "paths_gif", "trigger_gdf"],
            outputs="trigger_gif",
            name="create_animation"
        ),
        node(
            func=update_data_catalog_gif,
            inputs=[f"params:{key}", "trigger_gif", "paths_gif"],
            outputs=None,
            name="update_data_catalog_gif"
        )
    ],
    namespace=f"visualization.{value}",
    parameters={key:key})


def create_pipeline(**kwargs) -> Pipeline:
    conf_params = retrieve_global_parameters()
    
    child_pipelines = []
    
    for key, value in conf_params.items():
        if "visualization.date" in key and conf_params[key] is not None :
            child_pipelines.append(create_child_pipeline(key, value))
    
    visualization_pipeline = sum(child_pipelines)
    
    # --------- chain retrieve_global_params and visualization pipelines ---------
    mapping={}
    for e in visualization_pipeline.all_inputs():
        if "db_name" in e:
            mapping[e]="db_name"
            
    return pipeline(pipe=visualization_pipeline, inputs=mapping, tags=["visualization"])