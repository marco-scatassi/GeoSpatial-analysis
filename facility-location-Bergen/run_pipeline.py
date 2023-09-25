from kedro.runner.sequential_runner import SequentialRunner
from kedro.io import DataCatalog
from kedro.framework.startup import bootstrap_project
from kedro.framework.project import find_pipelines
from kedro_datasets.yaml import YAMLDataSet
import yaml
import sys
import time

project_path = r"\/Pund/Stab$/guest801981/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen"
metadata = bootstrap_project(project_path)

def initialize_session(pipeline_to_run):
    print(f"Running pipeline: {pipeline_to_run}")
    print("Loading runner...")
    runner = SequentialRunner()
    print("Finding pipelines...")
    pipelines = find_pipelines()
    
    pipeline_data_catalog_path = project_path + f"/conf/base/parameters/{pipeline_to_run}.yml"
    with open(pipeline_data_catalog_path, "r") as f:
        params = yaml.safe_load(f)
        
    return runner, pipelines, params

# ----------------------------- BUILD ADJACENCY MATRIX ------------------------------ #
def run_build_adjacency_matrix():
    pipeline_to_run = "build_adjacency_matrix"
    namespace_to_run = [".all_day0", ".morning0", ".midday0", ".afternoon0",
                        ".all_day1", ".morning1", ".midday1", ".afternoon1"]
    runner, pipelines, params = initialize_session(pipeline_to_run)
    
    for nms in namespace_to_run:
        print(f"Running namespace: {nms}")
        data_params_path = project_path + f"/conf/base/parameters/{pipeline_to_run}_TEMP.yml"
        params_data_set = YAMLDataSet(filepath=data_params_path)
        params[pipeline_to_run+nms]["time_of_day"] = nms
        params_data_set.save(params[pipeline_to_run+nms])
        data_catalog = DataCatalog(data_sets={f"params:{pipeline_to_run+nms}": params_data_set})
        runner.run(pipelines[pipeline_to_run].only_nodes_with_namespace(f"{pipeline_to_run+nms}"), data_catalog)
    

# ------------------------------ SOLUTION COMPARISON ------------------------------ #
def run_solution_comparison(fl_class="p-center"):
    pipeline_to_run = "solution_comparison"
    namespace_to_run = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
                            "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                            "20", "21", "22", "23", "24"]
    facility_numbers = [1,2,3]
    handpicked = [True, False] if fl_class == "p-center" else [True]
    runner, pipelines, params = initialize_session(pipeline_to_run)

    for fn in facility_numbers:
        for hp in handpicked:
            print(f"Running pipeline: {pipeline_to_run} with {fn} facilities and handpicked={hp}")
            for nms in namespace_to_run:
                print(f"Running namespace: {nms}")
                data_params_path = project_path + f"/conf/base/parameters/{pipeline_to_run}_TEMP.yml"
                params_data_set = YAMLDataSet(filepath=data_params_path)
                params[pipeline_to_run+nms]["facilities_number"] = fn
                params[pipeline_to_run+nms]["handpicked"] = hp
                params[pipeline_to_run+nms]["fl_class"] = fl_class
                params_data_set.save(params[pipeline_to_run+nms])
                data_catalog = DataCatalog(data_sets={f"params:{pipeline_to_run}{nms}": params_data_set})
                runner.run(pipelines[pipeline_to_run].only_nodes_with_namespace(f"{pipeline_to_run}.{nms}"), data_catalog)
            
# ------------------------------ FL DETERMINISTIC ------------------------------ #
def run_fl_deterministic(fl_class="p-center"):
    
    if fl_class not in ["p-center", "p-median"]:
        return Exception("Facility location class not recognized")
    
    pipeline_to_run = "fl_deterministic"
    namespace_to_run = [".data01", ".data02", ".data03"]#, ".data11", ".data12", ".data13"]
    if fl_class == "p-median":
        namespace_to_run = namespace_to_run[:3]
    
    runner, pipelines, params = initialize_session(pipeline_to_run)
    
    print(pipelines[pipeline_to_run])
    print(params)
    
    for nms in namespace_to_run:
        print(f"Running namespace: {nms}")
        data_params_path = project_path + f"/conf/base/parameters/{pipeline_to_run}_TEMP.yml"
        params_data_set = YAMLDataSet(filepath=data_params_path)
        params[pipeline_to_run+nms]["fl_class"] = fl_class
        params_data_set.save(params[pipeline_to_run+nms])
        data_catalog = DataCatalog(data_sets={f"params:{pipeline_to_run+nms}": params_data_set})
        runner.run(pipelines[pipeline_to_run].only_nodes_with_namespace(f"{pipeline_to_run+nms}"), data_catalog)
        time.sleep(10)

def main():    
    for i in range(1, len(sys.argv)):
        print(f"Argument {i}: {sys.argv[i]}")
    
    pipelines_to_run = []
    extra_params = {"fl_class": "p-center"}
    
    for i in range(1, len(sys.argv)):
        if sys.argv[i] in ["build_adjacency_matrix", "solution_comparison", "fl_deterministic"]:
            pipelines_to_run.append(sys.argv[i])
        elif sys.argv[i] in ["p-center", "p-median"]:
            extra_params["fl_class"] = sys.argv[i]
        else:
            raise Exception(f"Argument {i} not recognized")

    for pipeline_name in pipelines_to_run:
        if pipeline_name == "build_adjacency_matrix":
            run_build_adjacency_matrix()
        elif pipeline_name == "solution_comparison":
            run_solution_comparison(extra_params["fl_class"])
        elif pipeline_name == "fl_deterministic":
            run_fl_deterministic(extra_params["fl_class"])
        else:
            print("Pipeline name not recognized")
            return

    
if __name__ == "__main__":
    main()