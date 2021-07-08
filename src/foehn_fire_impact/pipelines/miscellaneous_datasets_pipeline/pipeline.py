"""
Pipeline which acts as a wrapper for two functions to load and preprocess additional datasets.
One will load spatial precipitation data, the other one fire indices data for each station.
"""
from kedro.pipeline import Pipeline, node
from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline([
        node(make_rain_dataset,
             ["foehn_stations", "parameters"],
             "rain_data",
             name="make_rain_dataset"
             ),
        node(load_fire_indices_data,
             ["foehn_stations", "params:regions"],
             "fire_indices_data",
             name="load_fire_indices_data"
             )
    ])
