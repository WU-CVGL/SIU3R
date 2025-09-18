import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from src.data.components.replica_dataset import ReplicaDataset
from src.data.config import DataLoaderCfg, DatasetCfg
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def collate_fn(examples):
    try:
        examples = list(filter(lambda x: x is not None, examples))
        if len(examples) == 0:
            raise ValueError("No valid examples found in the batch")
        context_views_images = np.array(
            [example["context_views_images"] for example in examples]
        )
        context_views_images = torch.tensor(context_views_images) / 255.0
        context_views_depths = np.array(
            [example["context_views_depths"] for example in examples]
        )
        context_views_depths = torch.tensor(context_views_depths, dtype=torch.float32)

        context_views_intrinsics = np.array(
            [example["context_views_intrinsics"] for example in examples]
        )
        context_views_intrinsics = torch.tensor(
            context_views_intrinsics, dtype=torch.float32
        )
        context_views_extrinsics = np.array(
            [example["context_views_extrinsics"] for example in examples]
        )
        context_views_extrinsics = torch.tensor(
            context_views_extrinsics, dtype=torch.float32
        )
        target_views_images = np.array(
            [example["target_views_images"] for example in examples]
        )
        target_views_images = torch.tensor(target_views_images) / 255.0
        target_views_depths = np.array(
            [example["target_views_depths"] for example in examples]
        )
        target_views_depths = torch.tensor(target_views_depths, dtype=torch.float32)
        target_views_intrinsics = np.array(
            [example["target_views_intrinsics"] for example in examples]
        )
        target_views_intrinsics = torch.tensor(
            target_views_intrinsics, dtype=torch.float32
        )
        target_views_extrinsics = np.array(
            [example["target_views_extrinsics"] for example in examples]
        )
        target_views_extrinsics = torch.tensor(
            target_views_extrinsics, dtype=torch.float32
        )

        context_mask_labels = [example["context_mask_labels"] for example in examples]
        context_class_labels = [example["context_class_labels"] for example in examples]
        target_mask_labels = [example["target_mask_labels"] for example in examples]
        target_class_labels = [example["target_class_labels"] for example in examples]
        scene_names = [example["scene_names"] for example in examples]
        context_views_id = [example["context_views_id"] for example in examples]
        target_views_id = [example["target_views_id"] for example in examples]
        # Return a dictionary of all the collated features
        return {
            "scene_names": scene_names,
            "context_views_id": context_views_id,
            "context_views_images": context_views_images,
            "context_views_depths": context_views_depths,
            "context_views_intrinsics": context_views_intrinsics,
            "context_views_extrinsics": context_views_extrinsics,
            "target_views_id": target_views_id,
            "target_views_images": target_views_images,
            "target_views_depths": target_views_depths,
            "target_views_intrinsics": target_views_intrinsics,
            "target_views_extrinsics": target_views_extrinsics,
            "context_mask_labels": context_mask_labels,
            "context_class_labels": context_class_labels,
            "target_mask_labels": target_mask_labels,
            "target_class_labels": target_class_labels,
        }
    except Exception as e:
        raise e


class ReplicaDataModule(LightningDataModule):
    def __init__(
        self,
        train_loader_cfg: DataLoaderCfg,
        val_loader_cfg: DataLoaderCfg,
        test_loader_cfg: DataLoaderCfg,
        dataset_cfg: DatasetCfg,
    ):
        super().__init__()
        self.train_dataloader_cfg = train_loader_cfg
        self.val_dataloader_cfg = val_loader_cfg
        self.test_dataloader_cfg = test_loader_cfg
        self.dataset_cfg = dataset_cfg
        self.save_hyperparameters(logger=False)

    def train_dataloader(self):
        return DataLoader(
            ReplicaDataset(
                root=self.dataset_cfg.data_dir,
                seg_task=self.dataset_cfg.seg_task,
                image_width=self.dataset_cfg.image_width,
                image_height=self.dataset_cfg.image_height,
                train=True,
                num_extra_context_views=self.dataset_cfg.num_extra_context_views,
                num_extra_target_views=self.dataset_cfg.num_extra_target_views,
                val_pair_json=self.dataset_cfg.val_pair_json,
            ),
            batch_size=self.train_dataloader_cfg.batch_size,
            num_workers=self.train_dataloader_cfg.num_workers,
            pin_memory=self.train_dataloader_cfg.pin_memory,
            collate_fn=collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            ReplicaDataset(
                root=self.dataset_cfg.data_dir,
                seg_task=self.dataset_cfg.seg_task,
                train=False,
                num_extra_context_views=self.dataset_cfg.num_extra_context_views,
                num_extra_target_views=self.dataset_cfg.num_extra_target_views,
                val_pair_json=self.dataset_cfg.val_pair_json,
            ),
            batch_size=self.val_dataloader_cfg.batch_size,
            num_workers=self.val_dataloader_cfg.num_workers,
            pin_memory=self.val_dataloader_cfg.pin_memory,
            collate_fn=collate_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            ReplicaDataset(
                root=self.dataset_cfg.data_dir,
                seg_task=self.dataset_cfg.seg_task,
                train=False,
                num_extra_context_views=self.dataset_cfg.num_extra_context_views,
                num_extra_target_views=self.dataset_cfg.num_extra_target_views,
                val_pair_json=self.dataset_cfg.val_pair_json,
            ),
            batch_size=self.test_dataloader_cfg.batch_size,
            num_workers=self.test_dataloader_cfg.num_workers,
            pin_memory=self.test_dataloader_cfg.pin_memory,
            collate_fn=collate_fn,
            shuffle=False,
        )
