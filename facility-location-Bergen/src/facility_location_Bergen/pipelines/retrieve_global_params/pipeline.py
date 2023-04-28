"""
This is a boilerplate pipeline 'retrieve_global_params'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import retrieve_db_name, retrieve_raw_data_root_dir


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=retrieve_db_name,
            inputs=None,
            outputs="db_name",
            name="retrieve_db_name"
        ),
        node(
            func=retrieve_raw_data_root_dir,
            inputs=None,
            outputs="raw_data_root_dir",
            name="retrieve_raw_data_root_dir"
        )
    ], tags=["retrieve_global_params"])
