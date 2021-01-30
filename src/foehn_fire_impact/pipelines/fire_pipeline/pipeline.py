"""
Pipeline for the fire data
"""

from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                cleanse_fire_data,
                "fire_data",
                "fire_data_cleansed",
                name="cleanse_fire_data"
            ),
            node(
                transform_datetime,
                "fire_data_cleansed",
                "fire_data_cleansed_datetime",
                name="transform_datetime"
            )
        ]
    )
