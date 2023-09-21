import os
from kedro.config import ConfigLoader
from kedro.framework.project import settings

# --------------------------------------------- retrieve_global_parameters ---------------------------------------------
def get_project_directory():
    # retrieve the current working directory
    cwd = os.getcwd()
    # traverse up the directory tree until you find the root folder of the project
    while not os.path.exists(os.path.join(cwd, "src")):
        cwd = os.path.dirname(cwd)
    # the root folder of the project is the parent directory of "src"
    project_dir = cwd
    return project_dir


def retrieve_catalog_path():
    project_dir = get_project_directory()
    return f"{project_dir}\\conf\\base\\catalog.yml"


def retrieve_global_parameters():
    project_dir = get_project_directory()
    conf_path = f"{project_dir}\\{settings.CONF_SOURCE}"
    conf_loader = ConfigLoader(conf_source=conf_path, env="local")
    conf_params = conf_loader["parameters"]
    return conf_params


def retrieve_db_name():
    conf_params = retrieve_global_parameters()
    return conf_params["db_name"]


def retrieve_raw_data_root_dir():
    conf_params = retrieve_global_parameters()
    return conf_params["raw_data_root_dir"]


def retrieve_gdf_path(date: dict, processed=False, subSegment=False):
    if type(date) == str:
        dt = date
    else:
        dt = date["day"]

    # define saving paths
    if processed and subSegment:
        saving_path = f"data/03_primary/{dt}_processed_subSegment.geojson"
    elif processed and not subSegment:
        saving_path = f"data/03_primary/{dt}_processed.geojson"
    else:
        saving_path = f"data/03_primary/{dt}.geojson"
    return saving_path


def retrieve_gdf_average_path(time, subSegment=False):
    if subSegment:
        saving_path = f"data/03_primary/average_{time}_subSegment.geojson"
    else:
        saving_path = f"data/03_primary/average_{time}.geojson"
    return saving_path


def retrieve_gdf_worst_average_path(time):
    saving_path = f"data/03_primary/worst_average_{time}.geojson"
    return saving_path


def retrieve_average_graph_path(time, connected=True, splitted=False, firstSCC=False, backslah=False):
    if connected:
        saving_path = rf"data/03_primary/average_graph_{time}_connected.pkl"
        if splitted:
            saving_path = rf"data/03_primary/average_graph_{time}_connected_splitted.pkl"
            if firstSCC:
                saving_path = rf"data/03_primary/average_graph_{time}_connected_splitted_firstSCC.pkl"
    else:
        saving_path = rf"data/03_primary/average_graph_{time}_original.pkl"
    if backslah:
        saving_path = saving_path.replace("/", "\\")
    return saving_path


def retrieve_worst_average_graph_path(time, connected=True, backslah=False):
    if connected:
        saving_path = rf"data/03_primary/worst_average_graph_{time}_connected.pkl"
    else:
        saving_path = rf"data/03_primary/worst_average_graph_{time}.pkl"
    if backslah:
        saving_path = saving_path.replace("/", "\\")
    return saving_path


def retrieve_adj_matrix_path(time, free_flow=False, handpicked=False):
    if handpicked:
        if free_flow:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/random_candidate_plus_handpicked/adj_matrix_{time}_free_flow.pkl"
        else:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/random_candidate_plus_handpicked/adj_matrix_{time}.pkl"
    else:
        if free_flow:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/only_random_candidate_location/adj_matrix_{time}_free_flow.pkl"
        else:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/only_random_candidate_location/adj_matrix_{time}.pkl"
    return saving_path


def retrieve_worst_adj_matrix_path(time, free_flow=False, handpicked=False):
    if handpicked:
        if free_flow:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/random_candidate_plus_handpicked/worst_adj_matrix_{time}_free_flow.pkl"
        else:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/random_candidate_plus_handpicked/worst_adj_matrix_{time}.pkl"
    else:
        if free_flow:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/only_random_candidate_location/worst_adj_matrix_{time}_free_flow.pkl"
        else:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/only_random_candidate_location/worst_adj_matrix_{time}.pkl"
    return saving_path


def retrieve_adj_mapping_path(time, free_flow=False, handpicked=False):
    if handpicked:
        if free_flow:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/random_candidate_plus_handpicked/adj_mapping_all_day.pkl"
        else:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/random_candidate_plus_handpicked/adj_mapping_{time}.pkl"
    else:
        if free_flow:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/only_random_candidate_location/adj_mapping_all_day.pkl"
        else:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/only_random_candidate_location/adj_mapping_{time}.pkl"
    return saving_path

def retrieve_adj_mapping_path_2(time, free_flow=False, handpicked=False):
    if handpicked:
        if free_flow:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/random_candidate_plus_handpicked/adj_mapping_2_all_day_free_flow.pkl"
        else:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/random_candidate_plus_handpicked/adj_mapping_2_{time}.pkl"
    else:
        if free_flow:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/only_random_candidate_location/adj_mapping_2_all_day_free_flow.pkl"
        else:
            saving_path = f"data/03_primary/adjacency_matrix_and_mapping/only_random_candidate_location/adj_mapping_2_{time}.pkl" 
    return saving_path

def retrieve_worst_adj_mapping_path(time):
    saving_path = f"data/03_primary/worst_adj_mapping_{time}.pkl"
    return saving_path

def retrieve_solution_path(facilities_number, time=None, stochastic=False, handpicked=False):
    if handpicked:
        if stochastic:
            path = f"data/07_model_output/random_candidate_plus_handpicked/{facilities_number}_locations/stochastic_solution/lshape_solution.pkl"
        else:
            path = f"data/07_model_output/random_candidate_plus_handpicked/{facilities_number}_locations/deterministic_exact_solutions/exact_solution_{time}.pkl"  
    else:
        if stochastic:
            path = f"data/07_model_output/only_random_candidate_location/{facilities_number}_locations/stochastic_solution/lshape_solution.pkl"
        else:
            path = f"data/07_model_output/only_random_candidate_location/{facilities_number}_locations/deterministic_exact_solutions/exact_solution_{time}.pkl"  
    return path

def retrieve_light_solution_path(facilities_number, time, handpicked=False):
    if handpicked:
        path = f"data/07_model_output/random_candidate_plus_handpicked/{facilities_number}_locations/deterministic_exact_solutions/light_exact_solution_{time}.pkl"
    else:
        path = f"data/07_model_output/only_random_candidate_location/{facilities_number}_locations/deterministic_exact_solutions/light_exact_solution_{time}.pkl"
    return path
    
def retrieve_solution_vs_scenario_path(facilities_number, time_solution, time_scenario, weight, 
                                       worst=False, handpicked=False):
    if handpicked:
        if worst:
            saving_path = f"data/08_reporting/random_candidate_plus_handpicked/{facilities_number}_locations/solution_vs_scenario_{time_solution}_{time_scenario}_{weight}_worst.pkl"
        else:
            saving_path = f"data/08_reporting/random_candidate_plus_handpicked/{facilities_number}_locations/solution_vs_scenario_{time_solution}_{time_scenario}_{weight}.pkl"
    else:
        if worst:
            saving_path = f"data/08_reporting/only_random_candidate_location/{facilities_number}_locations/solution_vs_scenario_{time_solution}_{time_scenario}_{weight}_worst.pkl"
        else:
            saving_path = f"data/08_reporting/only_random_candidate_location/{facilities_number}_locations/solution_vs_scenario_{time_solution}_{time_scenario}_{weight}.pkl"
    return saving_path


def retrieve_gif_saving_path(date: dict):
    # define saving paths
    saving_path = f"data/08_reporting/AnimatedPlot{date['day']}{date['time']}.gif"
    return saving_path
