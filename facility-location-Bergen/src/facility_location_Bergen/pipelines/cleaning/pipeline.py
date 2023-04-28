"""
This is a boilerplate pipeline 'cleaning'
generated using Kedro 0.18.7
"""

import sys
sys.path.append(r'C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules')

from .nodes import *
from kedro.pipeline import Pipeline, node, pipeline
from retrieve_global_parameters import retrieve_db_name, retrieve_global_parameters

def create_child_pipeline(key, value) -> list:
    return pipeline([
        node(
          func=verify_cleaning_already_done,  
          inputs=[f"params:{key}"],
          outputs="is_done",
          name="verify_cleaning_already_done"
        ),
        node(
            func=filter_data_geographically,
            inputs=["db_name", f"params:{key}", "params:polygon_vertex", f"trigger_{value}", "is_done"],
            outputs="trigger_filter",
            name="filter_data_geographically"
            ),
        node(
            func=keep_common_road_segments_across_time,
            inputs=["db_name", f"params:{key}", "trigger_filter", "is_done"],
            outputs="trigger_keep",
            name="keep_common_road_segments_across_time"
            ),
        node(
            func=remove_unnecessary_fields,
            inputs=["db_name", f"params:{key}", "trigger_keep", "is_done"],
            outputs="trigger_remove",
            name="remove_unnecessary_fields"
            ),
        node(
            func=update_data_catalog_trigger,
            inputs=["trigger_remove", f"params:{key}"],
            outputs=None,
            name="update_data_catalog_trigger"
        )
        ],
    namespace=f"cleaning.{value}",
    parameters={"polygon_vertex":"cleaning.polygon_vertex", key:key})


def create_pipeline(**kwargs) -> Pipeline:
    conf_params = retrieve_global_parameters()
    
    child_pipelines = []
    
    for key, value in conf_params.items():
        if "cleaning.date" in key and conf_params[key] is not None :
            child_pipelines.append(create_child_pipeline(key, value))
    
    cleaning_pipeline = sum(child_pipelines)
    
    # --------- chain retrieve_global_params and cleaning pipelines ---------
    mapping={}
    for e in cleaning_pipeline.all_inputs():
        if "db_name" in e:
            mapping[e]="db_name"
                
        if "raw_data_root_dir" in e:
            mapping[e]="raw_data_root_dir"  
        
    return pipeline(pipe=cleaning_pipeline, inputs=mapping, tags=["cleaning"])