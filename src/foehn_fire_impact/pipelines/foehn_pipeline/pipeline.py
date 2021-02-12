

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
                name="load_old_foehn_data"
            )
        ]
    )
