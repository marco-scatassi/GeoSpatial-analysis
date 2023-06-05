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
                func=create_worst_average_gdfs,
                inputs=[f"params:{key}"],
                outputs="trigger1",
                name="create_average_gdfs",
            ),
            node(
                func=update_data_catalog_gdf,
                inputs=[f"trigger1"],
                outputs="finished",
                name="update_data_catalog_gdf_average",
            ),
        ],
        namespace=f"build_worst_average_gdfs",
        parameters={key: key},
    )


def create_pipeline(**kwargs) -> Pipeline:
    conf_params = retrieve_global_parameters()

    child_pipelines = []

    for key, value in conf_params.items():
        if "build_worst_average_gdfs" in key:
            child_pipelines.append(create_child_pipeline(key, value))

    create_average_gdfs_pipeline = sum(child_pipelines)

    return pipeline(
        pipe=create_average_gdfs_pipeline, tags=["build_worst_average_gdfs"]
    )
