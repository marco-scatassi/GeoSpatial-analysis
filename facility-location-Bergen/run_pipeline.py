from kedro.runner.sequential_runner import SequentialRunner
from kedro.io import DataCatalog
from kedro.framework.startup import bootstrap_project
from kedro.framework.project import find_pipelines
from kedro_datasets.yaml import YAMLDataSet
import yaml

project_path = r"\/Pund/Stab$/guest801981/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen"
metadata = bootstrap_project(project_path)

def initialize_session(pipeline_to_run):
    print(f"Running pipeline: {pipeline_to_run}")
    print("Loading runner...")
    runner = SequentialRunner()
    print("Finding pipelines...")
    pipelines = find_pipelines()
    return runner,pipelines

# ----------------------------- BUILD ADJACENCY MATRIX ------------------------------ #
def run_build_adjacency_matrix():
    pipeline_to_run = "build_adjacency_matrix"
    namespace_to_run = ["all_day", "morning", "midday", "afternoon"]

    runner, pipelines = initialize_session(pipeline_to_run)
    


# ------------------------------ SOLUTION COMPARISON ------------------------------ #
def run_solution_comparison():
    pipeline_to_run = "solution_comparison"
    namespace_to_run = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
                            "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                            "20", "21", "22", "23", "24"]
    facility_numbers = [1,2,3]

    runner, pipelines = initialize_session(pipeline_to_run)

    pipeline_data_catalog_path = project_path + f"/conf/base/parameters/{pipeline_to_run}.yml"
    with open(pipeline_data_catalog_path, "r") as f:
        params = yaml.safe_load(f)

    for fn in facility_numbers:
        print(f"Running pipeline: {pipeline_to_run} with {fn} facilities")
        for i, nms in enumerate(namespace_to_run):
            print(f"Running namespace: {nms}")
            data_params_path = project_path + f"/conf/base/parameters/{pipeline_to_run}_TEMP.yml"
            params_data_set = YAMLDataSet(filepath=data_params_path)
            params[pipeline_to_run+nms]["facilities_number"] = fn
            params_data_set.save(params[pipeline_to_run+nms])
            data_catalog = DataCatalog(data_sets={f"params:{pipeline_to_run}{nms}": params_data_set})
            runner.run(pipelines[pipeline_to_run].only_nodes_with_namespace(f"{pipeline_to_run}.{nms}"), data_catalog)


