"""
This is a boilerplate pipeline 'new_collections_from_raw'
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
            func=compose_url_to_raw_data,
            inputs=[f"params:{key}", "raw_data_root_dir"],
            outputs="urls",
            name="compose_url_to_raw_data"
        ),
        node(
            func=insert_raw_data,
            inputs=["urls", "db_name", f"params:{key}"],
            outputs=None,
            name="insert_raw_data"
        ),
        node(
            func=process_raw_data,
            inputs=["urls", "db_name", f"params:{key}"],
            outputs="processed_data",
            name="process_raw_data"
        ),
        node(
            func=insert_processed_data,
            inputs=["processed_data", "db_name",f"params:{key}"],
            outputs="trigger",
            name="insert_documents_in_the_collections"
        ),
        node(
            func=update_data_catalog_trigger,
            inputs=["trigger", f"params:{key}"],
            outputs=f"finished_{value}",
            name="update_data_catalog_trigger"
        )
    ],
    namespace=f"ingestion.{value}",
    parameters={key:key})


def create_pipeline(**kwargs) -> Pipeline:
    conf_params = retrieve_global_parameters()
    
    child_pipelines = []
    
    for key, value in conf_params.items():
        if "ingestion.date" in key and conf_params[key] is not None :
            child_pipelines.append(create_child_pipeline(key, value))
    
    ingestion_pipeline = sum(child_pipelines)
    
    # --------- chain retrieve_global_params and ingestion pipelines ---------
    mapping={}
    for e in ingestion_pipeline.all_inputs():
        if "db_name" in e:
            mapping[e]="db_name"
                
        if "raw_data_root_dir" in e:
            mapping[e]="raw_data_root_dir"  
            
    return pipeline(pipe=ingestion_pipeline, inputs=mapping, tags=["ingestion"])
