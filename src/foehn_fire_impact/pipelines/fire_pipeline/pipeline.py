"""
Pipeline for the fire data. Cleanse the data, transform datetime, add missing coordinates, and map to closest
weather observation station.
"""

from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                cleanse_fire_data,
                "raw_fire_data",
                "fire_data_basic_cleansed",
                name="cleanse_fire_data",
                tags="setup"
            ),
            node(
                transform_datetime,
                "fire_data_basic_cleansed",
                "fire_data_with_date_info",
                name="transform_datetime",
                tags="setup"
            ),
            node(
                fill_missing_coordinates,
                "fire_data_with_date_info",
                "fire_data_cleansed",
                name="fill_missing_coordinates",
                tags="setup"
            ),
            node(
                calculate_closest_station,
                ["fire_data_cleansed", "foehn_stations", "parameters"],
                "fire_data_with_closest_station",
                name="calculate_closest_station",
                tags="setup"
            )

        ]
    )
