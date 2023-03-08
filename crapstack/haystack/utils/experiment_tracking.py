from typing import Optional, Any, Dict, Union

from abc import ABC, abstractmethod
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def flatten_dict(dict_to_flatten: dict, prefix: str = ""):
    flat_dict = {}
    for k, v in dict_to_flatten.items():
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, prefix + k + "_"))
        else:
            flat_dict[prefix + k] = v
    return flat_dict


class BaseTrackingHead(ABC):
    """
    Base class for tracking experiments.

    This class can be extended to implement custom logging backends like MLflow, WandB, or TensorBoard.
    """

    @abstractmethod
    def init_experiment(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False,
    ):
        raise NotImplementedError()

    @abstractmethod
    def track_metrics(self, metrics: Dict[str, Any], step: int):
        raise NotImplementedError()

    @abstractmethod
    def track_artifacts(self, dir_path: Union[str, Path], artifact_path: Optional[str] = None):
        raise NotImplementedError()

    @abstractmethod
    def track_params(self, params: Dict[str, Any]):
        raise NotImplementedError()

    @abstractmethod
    def end_run(self):
        raise NotImplementedError()



