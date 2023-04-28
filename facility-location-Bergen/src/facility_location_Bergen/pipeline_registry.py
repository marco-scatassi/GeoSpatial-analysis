"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = Pipeline([])
    
    for pipe in pipelines.values():
        # ------------- chain ingestion and cleaning pipelines ---------------
        finished = False
        for e in pipe.outputs():
            if "finished" in e:
                finished = True
        
        if finished:
            mapping = {}
            for e in pipe.outputs():
                mapping[e] = e.replace("finished", "trigger").replace("ingestion", "cleaning")    
            pipelines["__default__"] += pipeline(pipe=pipe, outputs=mapping)
        else:
            pipelines["__default__"] += pipe
    
    return pipelines
