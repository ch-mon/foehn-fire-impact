"""Project hooks."""
from typing import Any, Dict, Iterable, Optional

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.versioning import Journal

from foehn_fire_impact.pipelines import fire_pipeline as firepipeline
from foehn_fire_impact.pipelines import foehn_pipeline as foehnpipeline
from foehn_fire_impact.pipelines import foehn_fire_pipeline as foehnfirepipeline
from foehn_fire_impact.pipelines import miscellaneous_datasets_pipeline as miscpipeline

class ProjectHooks:
    @hook_impl
    def register_pipelines(self) -> Dict[str, Pipeline]:
        """Register the project's pipeline.

        Returns:
            A mapping from a pipeline name to a ``Pipeline`` object.

        """
        fire_pipeline = firepipeline.create_pipeline()
        foehn_pipeline = foehnpipeline.create_pipeline()
        foehn_fire_pipeline = foehnfirepipeline.create_pipeline()
        misc_pipeline = miscpipeline.create_pipeline()

        return {
            "fire_pipeline": fire_pipeline,
            "foehn_pipeline": foehn_pipeline,
            "foehn_fire_pipeline": foehn_fire_pipeline,
            "miscellaneous_datasets_pipeline": misc_pipeline,
            "main_pipeline": fire_pipeline+foehn_fire_pipeline,
            "__default__": fire_pipeline+foehn_pipeline+foehn_fire_pipeline+misc_pipeline,
        }

    @hook_impl
    def register_config_loader(self, conf_paths: Iterable[str]) -> ConfigLoader:
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )
