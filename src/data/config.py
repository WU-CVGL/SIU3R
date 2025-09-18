from dataclasses import dataclass
from abc import ABC
from typing import Literal


@dataclass
class DatasetCfg:
    name: str
    data_dir: str
    image_width: int
    image_height: int
    seg_task: Literal["panoptic", "instance", "refer"]
    num_extra_context_views: int
    num_extra_target_views: int
    val_pair_json: str


@dataclass
class DataLoaderCfg:
    batch_size: int
    num_workers: int
    pin_memory: bool
