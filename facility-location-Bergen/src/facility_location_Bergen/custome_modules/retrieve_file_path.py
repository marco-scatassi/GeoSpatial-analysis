def retrieve_gdf_path(date: dict):
    # define saving paths
    saving_path = f"data/03_primary/{date['day']}.geojson"
    return saving_path

def retrieve_gif_saving_path(date: dict):
    # define saving paths
    saving_path = f"data/08_reporting/AnimatedPlot{date['day']}{date['time']}.gif"
    return saving_path