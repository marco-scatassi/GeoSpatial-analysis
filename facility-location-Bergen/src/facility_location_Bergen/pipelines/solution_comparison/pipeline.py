"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.7
"""

import sys

sys.path.append(
    # r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
    r'\\Pund\Stab$\guest801996\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules'
)

from .nodes import *
from retrieve_global_parameters import *
from kedro.pipeline import Pipeline, node, pipeline


def create_child_pipeline(key, value) -> list:
    return pipeline(
        [
            node(
                func=verify_df_already_created,
                inputs=[f"params:{key}"],
                outputs="is_created",
            ),
            node(
                func=solution_vs_scenario,
                inputs=[f"params:{key}", "is_created"],
                outputs="pickle_file",
                name="solution_vs_scenario",
            ),
            node(
                func=update_data_catalog_gdf,
                inputs=[f"pickle_file"],
                outputs="finished",
                name="update_data_catalog",
            ),
        ],
        namespace=f"solution_comparison.{key[-2:]}",
        parameters={key: key},
    )


def create_pipeline(**kwargs) -> Pipeline:
    conf_params = retrieve_global_parameters()

    child_pipelines = []

    for key, value in conf_params.items():
        if "solution_comparison" in key:
            child_pipelines.append(create_child_pipeline(key, value))

    create_average_gdfs_pipeline = sum(child_pipelines)

    return pipeline(pipe=create_average_gdfs_pipeline, tags=["solution_comparison"])
