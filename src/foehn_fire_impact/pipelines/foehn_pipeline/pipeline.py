"""
Pipeline to load and cleanse two data dumps from MeteoSwiss.
"""

from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                load_comprehensive_foehn_data,
                ["foehn_parameters", "parameters"],
                "raw_foehn_data",
                name="load_comprehensive_foehn_data"
            ),
            node(
                load_older_north_foehn_data,
                ["parameters"],
                "raw_north_foehn_data_old",
                name="load_older_north_foehn_data"
            ),
            node(
                merge_old_and_new_foehn_data,
                ["raw_north_foehn_data_old", "raw_foehn_data"],
                "foehn_data_cleansed",
                name="merge_old_and_new_foehn_data"
            ),
            node(
                prepare_foehn_data_for_forest_fire_merge,
                "foehn_data_cleansed",
                "foehn_data_prepared",
                name="prepare_foehn_data_for_forest_fire_merge"
            )
        ]
    )
