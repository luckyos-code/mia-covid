from dataclasses import dataclass, field
from typing import Tuple, Dict, List

@dataclass(eq=True, frozen=False)
class commons:
    weights_path: str = 'weights'
    data_path: str = 'data'
    random_seed: int = 42
    epochs: int = 20
    learning_rate: float = 1e-3
    l2_clip: float = 1.0
    mia_samplenb: int = 100

@dataclass(eq=True, frozen=False)
class covid: #class covid(commons):
    dataset_name: str = 'covid'
    img_shape: Tuple[int, int, int] = (224, 224, 3) # original size is 299x299x3
    val_test_split: Tuple[int, int] = (0.05, 0.15) # 80-5-15 for train-val-test
    imbalance_ratio: float = 1.5 # undersampling to 1.5x the normal images compared to covid
    variants: List[Dict] = field(default_factory=lambda: [
            {'activation': 'relu', 'pretraining': None},
            {'activation': 'relu', 'pretraining': 'imagenet'},
            {'activation': 'relu', 'pretraining': 'pneumonia'},
            {'activation': 'tanh', 'pretraining': None},
            {'activation': 'tanh', 'pretraining': 'imagenet'},
            {'activation': 'tanh', 'pretraining': 'pneumonia'},
        ])

@dataclass(eq=True, frozen=False)
class mnist:
    dataset_name: str = 'mnist'
    img_shape: Tuple[int, int, int] = (28, 28, 3) # original size is 28x28x1
    variants: List[Dict] = field(default_factory=lambda: [
            {'activation': 'relu', 'pretraining': None},
            {'activation': 'relu', 'pretraining': 'imagenet'},
            {'activation': 'tanh', 'pretraining': None},
            {'activation': 'tanh', 'pretraining': 'imagenet'},
        ])
        
#TODO class pneumonia

@dataclass(eq=True, frozen=False)
class resnet18:
    architecture: str = 'resnet18'
    batch_size: int = 32
    batch_size_private_covid: int = 16 # selection handled in code

@dataclass(eq=True, frozen=False)
class resnet50:
    architecture: str = 'resnet50'
    batch_size: int = 32
    batch_size_private_covid: int = 8 # selection handled in code
