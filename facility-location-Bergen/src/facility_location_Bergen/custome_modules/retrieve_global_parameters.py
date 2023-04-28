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