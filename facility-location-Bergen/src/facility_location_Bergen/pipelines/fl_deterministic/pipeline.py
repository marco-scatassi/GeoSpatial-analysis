"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.7
"""

import sys

sys.path.append(
    # r"C:\Users\Marco\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules"
    r'\\Pund\Stab$\guest801981\Documents\GitHub\GeoSpatial-analysis\facility-location-Bergen\src\facility_location_Bergen\custome_modules'
)

from .nodes import *
from retrieve_global_parameters import *
from kedro.pipeline import Pipeline, node, pipeline


def create_child_pipeline(key) -> list:
    return pipeline(
        [
            node(
              func=verify_problem_already_solved,
              inputs=[f"params:{key}"],
              outputs="already_solved",
              name="verify_problem_already_solved",  
            ),
            node(
                func=set_up_fl_problems,
                inputs=[f"params:{key}", "already_solved"],
                outputs="fls_exact",
                name="set_up_fl_problems",
            ),
            node(
                func=solve_fl_problems,
                inputs=[f"fls_exact", f"params:{key}"],
                outputs="message",
                name="solve_fl_problems",
            ),
        ],
        namespace=f"fl_deterministic",
        parameters={key: key},
    )


def create_pipeline(**kwargs) -> Pipeline:
    conf_params = retrieve_global_parameters()

    child_pipelines = []

    for key, value in conf_params.items():
        if "fl_deterministic" in key:
            child_pipelines.append(create_child_pipeline(key))

    complete_pipeline = sum(child_pipelines)

    return pipeline(pipe=complete_pipeline, tags=["fl_deterministic"])
