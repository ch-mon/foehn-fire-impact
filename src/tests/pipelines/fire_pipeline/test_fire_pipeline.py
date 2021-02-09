from src.foehn_fire_impact.pipelines.fire_pipeline.nodes import fill_missing_coordinates
from src.foehn_fire_impact.pipelines.fire_pipeline.utils import decimalWSG84_to_LV3, LV3_to_decimalWSG84
import numpy as np
from kedro.io import DataCatalog


class TestFirePipeline:
    def test_decimalWSG84_to_LV3(self):
        # Numbers are from SwissTopo
        x_fun, y_fun = decimalWSG84_to_LV3(lon=8.730499, lat=46.044127)

        assert np.round(x_fun, 0) == 100000
        assert np.round(y_fun, 0) == 700000

    def test_LV3_to_decimalWSG84(self):
        lon_fun, lat_fun = LV3_to_decimalWSG84(x = 100000, y=700000)

        assert np.round(lon_fun, 6) == 8.730499
        assert np.round(lat_fun, 6) == 46.044127

    def test_fill_missing_coordinates(self):
        io = DataCatalog.from_config(catalog="conf/base/catalog.yml")
        print(fill_missing_coordinates(io.load("fire_data_with_date_info")))