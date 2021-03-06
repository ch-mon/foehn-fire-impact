from src.foehn_fire_impact.pipelines.fire_pipeline.nodes import fill_missing_coordinates
from src.foehn_fire_impact.pipelines.fire_pipeline.utils import decimalWSG84_to_LV3, LV3_to_decimalWSG84
import numpy as np
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from numpy.testing import assert_allclose

class TestFirePipeline:
    def test_decimalWSG84_to_LV3(self):
        # Numbers are from SwissTopo
        x_fun, y_fun = decimalWSG84_to_LV3(lon=8.730499, lat=46.044127)

        assert np.round(x_fun, 0) == 100000
        assert np.round(y_fun, 0) == 700000

    def test_decimalWSG84_to_LV3_2(self):
        conf_loader = ConfigLoader(['conf/base'])
        conf_catalog = conf_loader.get('catalog*', 'catalog/**')
        catalog = DataCatalog.from_config(conf_catalog)
        df = catalog.load("foehn_stations")
        x_fun, y_fun = decimalWSG84_to_LV3(lon=df["longitude"], lat=df["latitude"])

        # There are somtimes deviations up to 1000m in the data from MeteoSwiss, thus higher atol
        assert_allclose(x_fun, df["x_LV03"], atol=1000)
        assert_allclose(y_fun, df["y_LV03"], atol=1000)

    def test_LV3_to_decimalWSG84(self):
        lon_fun, lat_fun = LV3_to_decimalWSG84(x = 100000, y=700000)

        assert np.round(lon_fun, 6) == 8.730499
        assert np.round(lat_fun, 6) == 46.044127

    def test_LV3_to_decimalWSG84_2(self):
        conf_loader = ConfigLoader(['conf/base'])
        conf_catalog = conf_loader.get('catalog*', 'catalog/**')
        catalog = DataCatalog.from_config(conf_catalog)
        df = catalog.load("foehn_stations")
        lon_fun, lat_fun = LV3_to_decimalWSG84(x=df["x_LV03"], y=df["y_LV03"])

        # There are some deviations in the data from MeteoSwiss, thus higher atol
        assert_allclose(lon_fun, df["longitude"], atol=0.01)
        assert_allclose(lat_fun, df["latitude"], atol=0.01)

    def test_fill_missing_coordinates(self):
        conf_loader = ConfigLoader(['conf/base'])
        conf_catalog = conf_loader.get('catalog*', 'catalog/**')
        catalog = DataCatalog.from_config(conf_catalog)
        df = catalog.load("fire_data_cleansed")

        assert df[["coordinates_x", "coordinates_y", "longitude", "latitude", "municipality"]].isnull().sum().sum() == 0