from dataclasses import dataclass, field
from typing import Literal, TypeVar, Type
import hydra
import os
from omegaconf import DictConfig, OmegaConf, open_dict
from dacite import from_dict
from src.data.config import DatasetCfg, DataLoaderCfg
from src.utils.scannet_constant import (
    PANOPTIC_SEMANTIC2NAME,
    INSTANCE_SEMANTIC2NAME,
    STUFF_CLASSES,
    THING_CLASSES,
)
from src.utils.coco_constant import (
    ADE20K_PANOPTIC_SEMANTIC2NAME,
    ADE20K_INSTANCE_SEMANTIC2NAME,
    ADE20K_STUFF_CLASSES,
    ADE20K_THING_CLASSES,
    COCO_PANOPTIC_SEMANTIC2NAME,
    COCO_INSTANCE_SEMANTIC2NAME,
    COCO_STUFF,
    COCO_THINGS,
)


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_epochs: int


@dataclass
class TrainerCfg:
    max_epochs: int
    accelerator: Literal["gpu", "cpu"]
    strategy: Literal["ddp", "ddp_find_unused_parameters_true"]
    devices: int
    accumulate_grad_batches: int
    gradient_clip_val: float
    check_val_every_n_epoch: int
    log_every_n_steps: int
    skip_sanity_check: bool
    precision: Literal["32", "16-mixed", "bf16-mixed"]


@dataclass
class CrocoCfg:
    enc_depth: int = 24
    dec_depth: int = 12
    enc_embed_dim: int = 1024
    dec_embed_dim: int = 768
    enc_num_heads: int = 16
    dec_num_heads: int = 12
    pos_embed: str = "RoPE100"
    patch_size: int = 16
    freeze: str = "encoder"


@dataclass
class Mask2formerCfg:
    id2label: dict[int, str] = field(default_factory=dict)
    seg_threshold: float = 0.5
    label_ids_to_fuse: list[int] = field(default_factory=list)
    num_queries: int = 100


@dataclass
class GaussianHeadCfg:
    gaussian_scale_min: float = 0.5
    gaussian_scale_max: float = 15.0
    sh_degree: int = 4


@dataclass
class ModelCfg:
    croco: CrocoCfg
    mask2former: Mask2formerCfg
    gaussian_head: GaussianHeadCfg
    image_size: list[int]
    pretrained_weights_path: str | None = None


@dataclass
class VisualizerCfg:
    log_colored_depth: bool
    log_rendered_video: bool
    log_gaussian_ply: bool
    save_sh_dc_only: bool
    dataset_name: str
    overlay_mask_alpha: float
    write_to: str


@dataclass
class EvaluatorCfg:
    dataset_name: str
    eval_context_miou: bool = True
    eval_context_pq: bool = True
    eval_context_map: bool = True
    eval_target_miou: bool = True
    eval_target_pq: bool = True
    eval_target_map: bool = True
    eval_image_quality: bool = True
    eval_depth_quality: bool = True
    id2label: dict[int, str] = field(default_factory=dict)
    stuffs: list[int] = field(default_factory=list)
    things: list[int] = field(default_factory=list)
    device: Literal["cpu", "cuda"] = "cuda"
    eval_path: str | None = None


@dataclass
class PipelineCfg:
    log_training_result_interval: int
    pretrained_weights_path: str
    weight_seg_loss: float
    enable_instance_depth_smoothness: bool
    weight_depth_smoothness: float
    model: ModelCfg
    visualizer: VisualizerCfg
    evaluator: EvaluatorCfg


@dataclass
class DatamoduleCfg:
    dataset_cfg: DatasetCfg
    train_loader_cfg: DataLoaderCfg
    val_loader_cfg: DataLoaderCfg
    test_loader_cfg: DataLoaderCfg


@dataclass
class RootCfg:
    trainer: TrainerCfg
    optimizer: OptimizerCfg
    datamodule: DatamoduleCfg
    pipeline: PipelineCfg
    project: str
    experiment: str
    wandb_mode: Literal["online", "offline"] = "offline"
    output_path: str | None = None
    ckpt_path: str | None = None
    mode: Literal["train", "test", "val"] = "train"
    seed: int = 0
    ignore_warnings: bool = True


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
) -> T:
    return from_dict(data_class, OmegaConf.to_container(cfg))


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    bind_cfg(cfg)
    return load_typed_config(
        cfg,
        RootCfg,
    )


def bind_cfg(cfg: DictConfig):
    with open_dict(cfg):
        cfg.output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        cfg.pipeline.model.image_size = (
            cfg.datamodule.dataset_cfg.image_height,
            cfg.datamodule.dataset_cfg.image_width,
        )
        cfg.pipeline.model.pretrained_weights_path = (
            cfg.pipeline.pretrained_weights_path
        )
        cfg.pipeline.visualizer.write_to = cfg.output_path
        cfg.pipeline.visualizer.dataset_name = cfg.datamodule.dataset_cfg.name
        cfg.pipeline.evaluator.dataset_name = cfg.datamodule.dataset_cfg.name

        if cfg.mode == "val" or cfg.mode == "test":
            cfg.datamodule.dataset_cfg.num_extra_target_views = 4
        if cfg.datamodule.dataset_cfg.name == "ade20k":
            cfg.pipeline.model.mask2former.id2label = ADE20K_PANOPTIC_SEMANTIC2NAME
            cfg.pipeline.model.mask2former.label_ids_to_fuse = ADE20K_STUFF_CLASSES
            cfg.pipeline.evaluator.id2label = ADE20K_PANOPTIC_SEMANTIC2NAME
            cfg.pipeline.evaluator.stuffs = ADE20K_STUFF_CLASSES
            cfg.pipeline.evaluator.things = ADE20K_THING_CLASSES
        elif cfg.datamodule.dataset_cfg.name == "coco":
            cfg.pipeline.model.mask2former.id2label = COCO_PANOPTIC_SEMANTIC2NAME
            cfg.pipeline.model.mask2former.label_ids_to_fuse = COCO_STUFF
            cfg.pipeline.evaluator.id2label = COCO_PANOPTIC_SEMANTIC2NAME
            cfg.pipeline.evaluator.stuffs = COCO_STUFF
            cfg.pipeline.evaluator.things = COCO_THINGS
        elif cfg.datamodule.dataset_cfg.seg_task == "panoptic":
            cfg.pipeline.model.mask2former.id2label = PANOPTIC_SEMANTIC2NAME
            cfg.pipeline.model.mask2former.label_ids_to_fuse = STUFF_CLASSES
            cfg.pipeline.evaluator.id2label = PANOPTIC_SEMANTIC2NAME
            cfg.pipeline.evaluator.stuffs = STUFF_CLASSES
            cfg.pipeline.evaluator.things = THING_CLASSES
