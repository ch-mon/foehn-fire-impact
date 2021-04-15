"""
Pipeline merging the before preprocessed fire and foehn data and adding some control variables (fire regime,
foehn type, and decade) .
"""
from kedro.pipeline import Pipeline, node
from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline([
        node(make_rain_dataset,
             ["foehn_stations", "parameters"],
             "rain_data",
             name="make_rain_dataset"
             )
    ])
