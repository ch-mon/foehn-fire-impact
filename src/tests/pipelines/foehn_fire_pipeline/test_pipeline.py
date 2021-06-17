from kedro.config import ConfigLoader
from kedro.framework.session import get_current_session
from kedro.io import DataCatalog


class TestFoehnFirePipeline:
    def test_final_dataset(self):

        catalog = DataCatalog.from_config(ConfigLoader('conf/base').get('catalog*', 'catalog/**'))
        df_fires = catalog.load("fire_data_with_foehn_and_control_variables")
        stations_ds = df_fires["abbreviation"].unique()

        regions = ConfigLoader('conf/base').get("parameters*")["regions"]
        stations_conf = []
        for reg, stations in regions.items():
            stations_conf.extend(stations)

        assert sorted(stations_ds) == sorted(stations_conf)
