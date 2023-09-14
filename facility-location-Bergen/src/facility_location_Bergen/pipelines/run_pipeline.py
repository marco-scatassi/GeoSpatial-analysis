from kedro.framework.session import KedroSession
from kedro.runner.sequential_runner import SequentialRunner
from kedro.io import DataCatalog
from kedro.framework.startup import bootstrap_project
from kedro.framework.project import find_pipelines

project_path = r"\/Pund/Stab$/guest801981/Documents/GitHub/GeoSpatial-analysis/facility-location-Bergen"
sequence_name_to_run = ["all_day", "morning", "midday", "afternoon"]
metadata = bootstrap_project(project_path)

runner = SequentialRunner()
catalog = DataCatalog()

pipelines = find_pipelines()


# session = KedroSession.create()
# pipeline_name = 'build_adjacency_matrix'
# pipeline = session.load_pipeline(pipeline_name)

# runner.run(pipeline, catalog)