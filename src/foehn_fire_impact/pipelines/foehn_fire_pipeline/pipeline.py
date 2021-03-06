"""
Pipeline merging the before preprocessed fire and foehn data and adding some control variables (fire regime,
foehn type, and decade) .
"""
from kedro.pipeline import Pipeline, node
from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline([
        node(map_fires_to_foehn,
             ["fire_data_cleansed_stations", "foehn_data_prepared"],
             "fire_data_with_foehn",
             name="map_fires_to_foehn"
             ),
        node(add_control_variables,
             "fire_data_with_foehn",
             "fire_data_with_foehn_and_control_variables",
             name="add_control_variables"
             )
    ])
