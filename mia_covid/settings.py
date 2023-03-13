from dataclasses import dataclass
from typing import Optional
from mia_covid.abstract_dataset_handler import AbstractDataset

from mia_covid.models import model_settings


# Settings
@dataclass(eq=True, frozen=False)
class privacy():
    target_eps: float
    delta: Optional[float] = None  # placeholder to set in model compilation
    noise: float = 0.0  # update in model compilation if private


@dataclass(eq=True, frozen=False)
class commons:
    weights_path: str = 'weights'
    data_path: str = 'data'
    random_seed: int = 42
    epochs: int = 20
    learning_rate: float = 1e-3
    l2_clip: float = 1.0
    mia_samplenb: int = 100


# TODO: - we could also use inheritance here, but composition seems like the cleaner aproach to me
#       - including the complex abstractdataset class in this composition is not really nice -> maybe split the abstractdataset class up into dataclass_settings and functionality class
@dataclass(eq=True, frozen=False)
class run_settings:
    commons: commons
    privacy: privacy
    model_setting: model_settings
    dataset: AbstractDataset


# TODO class pneumonia
