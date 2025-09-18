from src.data.config import DatasetCfg, DataLoaderCfg


def get_datamodule(
    dataset_cfg: DatasetCfg,
    train_loader_cfg: DataLoaderCfg,
    val_loader_cfg: DataLoaderCfg,
    test_loader_cfg: DataLoaderCfg,
):
    if dataset_cfg.name == "scannet":
        from src.data.datamodules.scannet_datamodule import ScanNetDataModule

        return ScanNetDataModule(
            train_loader_cfg=train_loader_cfg,
            val_loader_cfg=val_loader_cfg,
            test_loader_cfg=test_loader_cfg,
            dataset_cfg=dataset_cfg,
        )
    elif dataset_cfg.name == "replica":
        from src.data.datamodules.replica_datamodule import ReplicaDataModule

        return ReplicaDataModule(
            train_loader_cfg=train_loader_cfg,
            val_loader_cfg=val_loader_cfg,
            test_loader_cfg=test_loader_cfg,
            dataset_cfg=dataset_cfg,
        )
    elif dataset_cfg.name == "scannetpp":
        from src.data.datamodules.scannetpp_datamodule import ScanNetPPDataModule

        return ScanNetPPDataModule(
            train_loader_cfg=train_loader_cfg,
            val_loader_cfg=val_loader_cfg,
            test_loader_cfg=test_loader_cfg,
            dataset_cfg=dataset_cfg,
        )
    elif dataset_cfg.name == "concat":
        from src.data.datamodules.concat_datamodule import ConcatDataModule

        return ConcatDataModule(
            train_loader_cfg=train_loader_cfg,
            val_loader_cfg=val_loader_cfg,
            test_loader_cfg=test_loader_cfg,
            dataset_cfg=dataset_cfg,
        )
    elif dataset_cfg.name == "scanrefer":
        from src.data.datamodules.scanrefer_datamodule import ScanReferDataModule

        return ScanReferDataModule(
            train_loader_cfg=train_loader_cfg,
            val_loader_cfg=val_loader_cfg,
            test_loader_cfg=test_loader_cfg,
            dataset_cfg=dataset_cfg,
        )
    elif dataset_cfg.name == "ade20k":
        from src.data.datamodules.cocoformat_datamodule import ADE20KDataModule

        return ADE20KDataModule(
            train_loader_cfg=train_loader_cfg,
            val_loader_cfg=val_loader_cfg,
            test_loader_cfg=test_loader_cfg,
            dataset_cfg=dataset_cfg,
        )
    elif dataset_cfg.name == "coco":
        from src.data.datamodules.cocoformat_datamodule import COCODataModule

        return COCODataModule(
            train_loader_cfg=train_loader_cfg,
            val_loader_cfg=val_loader_cfg,
            test_loader_cfg=test_loader_cfg,
            dataset_cfg=dataset_cfg,
        )
    else:
        raise NotImplementedError(
            f"Dataset {dataset_cfg.name} not implemented. Please implement it in src/data/datamodules."
        )
