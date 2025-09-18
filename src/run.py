from typing import Any, Dict, List, Optional, Tuple
import hydra
import lightning as L
import warnings
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.config import load_typed_root_config, RootCfg
from src.data.get_datamodule import get_datamodule
from src.pipeline import Pipeline
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="main",
)
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    cfg: RootCfg = load_typed_root_config(cfg)
    if cfg.ignore_warnings:
        warnings.filterwarnings("ignore")
    L.seed_everything(cfg.seed, workers=True)

    mode = cfg.mode
    log.info(f"Running in {mode} mode")
    log.info(f"Config: {cfg}")

    wandb_logger = WandbLogger(
        project=cfg.project,
        name=cfg.experiment,
        offline=cfg.wandb_mode != "online",
        save_dir=cfg.output_path,
    )
    log.info(f"Logging to wandb: {wandb_logger.experiment.get_url()}")

    ckpt_path = cfg.ckpt_path
    if mode == "train" and ckpt_path is not None:
        log.info(f"training resuming from checkpoint: {ckpt_path}")
    elif mode == "test" and ckpt_path is None:
        log.error("No checkpoint path provided for testing. Aborted.")
        raise ValueError("No checkpoint path provided for testing. Aborted.")
    elif mode == "val" and ckpt_path is None:
        log.error("No checkpoint path provided for validation. Aborted.")
        raise ValueError("No checkpoint path provided for validation. Aborted.")

    log.info("Instantiating datamodule...")
    datamodule = get_datamodule(
        dataset_cfg=cfg.datamodule.dataset_cfg,
        train_loader_cfg=cfg.datamodule.train_loader_cfg,
        val_loader_cfg=cfg.datamodule.val_loader_cfg,
        test_loader_cfg=cfg.datamodule.test_loader_cfg,
    )

    log.info("Instantiating pipeline...")
    pipeline: Pipeline = Pipeline(cfg)

    callbacks = [
        RichModelSummary(max_depth=2),
        ModelCheckpoint(
            dirpath=f"{cfg.output_path}/checkpoints",
            filename="{epoch:03d}-{step}",
            every_n_epochs=cfg.trainer.check_val_every_n_epoch,
            save_on_train_epoch_end=True,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    log.info("Instantiating trainer...")
    trainer: Trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        num_sanity_val_steps=0 if cfg.trainer.skip_sanity_check else 2,
        callbacks=callbacks,
        default_root_dir=cfg.output_path,
        logger=wandb_logger,
    )

    if mode == "train":
        log.info("Starting training!")
        trainer.fit(model=pipeline, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info("Training finished!")
    elif mode == "test":
        log.info("Starting testing!")
        trainer.test(model=pipeline, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info("Testing finished!")
    elif mode == "val":
        log.info("Starting validation!")
        trainer.validate(model=pipeline, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info("Validation finished!")


if __name__ == "__main__":
    main()
