"""
This is a boilerplate pipeline 'new_collections_from_raw'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.config import ConfigLoader
from kedro.framework.project import settings
from .nodes import *

def create_child_pipeline(key) -> list:
    return pipeline([
        node(
            func=compose_url_to_raw_data,
            inputs=["params:db_name", f"params:{key}", "params:root_dir"],
            outputs="urls",
            name="compose_url_to_raw_data"
        ),
        node(
            func=insert_raw_data,
            inputs=["urls", "params:db_name", f"params:{key}"],
            outputs=None,
            name="insert_raw_data"
        ),
        node(
            func=process_raw_data,
            inputs=["urls"],
            outputs="processed_data",
            name="process_raw_data"
        ),
        node(
            func=insert_processed_data,
            inputs=["processed_data", "params:db_name",f"params:{key}"],
            outputs=None,
            name="insert_documents_in_the_collections"
        )
    ],
    namespace=f"{key}",
    parameters={"db_name":"db_name", key:key, "root_dir":"root_dir"})


def create_pipeline(**kwargs) -> Pipeline:
    conf_path = f"C:\\Users\\Marco\\Documents\\GitHub\\GeoSpatial-analysis\\facility-location-Bergen\\{settings.CONF_SOURCE}"
    conf_loader = ConfigLoader(conf_source=conf_path, env="local")
    conf_params = conf_loader["parameters"]
    
    child_pipelines = []
    
    for key in conf_params.keys():
        if "date" in key and conf_params[key] is not None :
            child_pipelines.append(create_child_pipeline(key))
    
    return sum(child_pipelines)
