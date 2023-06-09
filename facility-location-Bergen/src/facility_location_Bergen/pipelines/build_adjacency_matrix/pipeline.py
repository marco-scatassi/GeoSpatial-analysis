"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.7
"""

import sys

sys.path.append(
    r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
)

from .nodes import *
from retrieve_global_parameters import *
from kedro.pipeline import Pipeline, node, pipeline


def create_child_pipeline(key, value) -> list:
    return pipeline(
        [
            node(
                func=build_adj_matrix,
                inputs=[f"params:{key}"],
                outputs="trigger",
                name="build_adj_matrix",
            ),
            node(
                func=update_data_catalog,
                inputs=[f"params:{key}", "trigger"],
                outputs=None,
                name="update_data_catalog",
            ),
        ],
        namespace=f"build_adjacency_matrix.{key}",
        parameters={key: key},
    )


def create_pipeline(**kwargs) -> Pipeline:
    conf_params = retrieve_global_parameters()

    child_pipelines = []

    for key, value in conf_params.items():
        if "build_adjacency_matrix" in key:
            child_pipelines.append(create_child_pipeline(key, value))

    build_adjacency_matrix_pipeline = sum(child_pipelines)

    return pipeline(
        pipe=build_adjacency_matrix_pipeline, tags=["build_adjacency_matrix"]
    )
