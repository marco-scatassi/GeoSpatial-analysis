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

def retrieve_gdf_path(date: dict, processed=False):
    if type(date) == str:
        dt = date
    else:
        dt = date['day']
        
    # define saving paths
    if processed:
        saving_path = f"data/03_primary/{dt}_processed.geojson"
    else:
        saving_path = f"data/03_primary/{dt}.geojson"
    return saving_path

def retrieve_gdf_average_path(time):
    saving_path = f"data/03_primary/average_{time}.geojson"
    return saving_path

def retrieve_average_graph_path(time):
    saving_path = f"data/03_primary/average_graph_{time}.pkl"
    return saving_path

def retrieve_gif_saving_path(date: dict):
    # define saving paths
    saving_path = f"data/08_reporting/AnimatedPlot{date['day']}{date['time']}.gif"
    return saving_path