import os
import os.path as osp
import json
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.models.mask2former import VideoMask2FormerImageProcessor
from src.utils.scannet_constant import (
    PANOPTIC_SEMANTIC2NAME,
    PANOPTIC_SEMANTIC2CONTINUOUS,
    INSTANCE_SEMANTIC2CONTINUOUS,
    STUFF_CLASSES,
)
from src.utils.tensor_utils import inspect_shape
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ScanNetPPDataset(Dataset):
    def __init__(
        self,
        root: str,
        num_extra_context_views: int = 2,
        num_extra_target_views: int = 2,
        image_width: int = 256,
        image_height: int = 256,
        train: bool = True,
        seg_task: str = "panoptic",
        val_pair_json: str = "val_pair.json",
    ):
        super().__init__()
        self.root = root
        self.num_extra_context_views = num_extra_context_views
        self.num_extra_target_views = num_extra_target_views
        self.seg_task = seg_task
        self.train = train
        if self.train:
            self.scans_dir = osp.join(self.root, "train")
        else:
            self.scans_dir = osp.join(self.root, "val")
            if "demo" in val_pair_json:
                self.scans_dir = osp.join(self.root, "train")
            with open(osp.join(self.root, val_pair_json)) as f:
                self.val_pairs = json.load(f)
        scan_names = os.listdir(self.scans_dir)
        self.scan_names = [
            scan_name
            for scan_name in scan_names
            if osp.isdir(osp.join(self.scans_dir, scan_name))
        ]
        self.scan_names = sorted(self.scan_names)
        self.scan_items = {
            scan_name: sorted(
                [
                    int(item.split(".")[0])
                    for item in os.listdir(osp.join(self.scans_dir, scan_name, "depth"))
                ]
            )
            for scan_name in self.scan_names
        }

        self.processor = VideoMask2FormerImageProcessor(
            size=(256, 256),
            do_resize=False,
            reduce_labels=True,
            do_rescale=False,
            do_normalize=False,
            ignore_index=255,
            num_labels=20,
        )

    def __len__(self) -> int:
        return len(self.scan_names) if self.train else len(self.val_pairs)

    def intrinsics_normalize(self, intrinsics: list[np.ndarray]) -> list[np.ndarray]:
        # the first row is divided by image width, and the second row is divided by image height
        return [
            np.array(
                [
                    [intrinsics[0][0] / 256, 0, intrinsics[0][2] / 256],
                    [0, intrinsics[1][1] / 256, intrinsics[1][2] / 256],
                    [0, 0, 1],
                ]
            )
            for intrinsics in intrinsics
        ]

    def relative_pose(
        self,
        context_views_extrinsics: list[np.ndarray],
        target_views_extrinsics: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        calculate relative poses, make the first context view extrinsic as the canonical frame

        Args:
            context_views_extrinsics: [4, 4] * 2
            target_views_extrinsics: [4, 4] * N
        Returns:
            relative_context_views_extrinsics: [4, 4] * 2
            relative_target_views_extrinsics: [4, 4] * N
        """
        # make the first context view extrinsic as the canonical frame
        canonical_frame = context_views_extrinsics[0]
        canonical_inv = np.linalg.inv(canonical_frame)
        relative_context_views_extrinsics = [
            canonical_inv @ extrinsic for extrinsic in context_views_extrinsics
        ]
        relative_target_views_extrinsics = [
            canonical_inv @ extrinsic for extrinsic in target_views_extrinsics
        ]
        return relative_context_views_extrinsics, relative_target_views_extrinsics

    def __getitem__(self, idx: int):
        try:
            if self.train:
                scan_name = self.scan_names[idx]
                scan_path = osp.join(self.scans_dir, scan_name)
                items = self.scan_items[scan_name]
                iou = torch.load(
                    osp.join(self.scans_dir, scan_name, "iou.pt"), weights_only=True
                )
                flag = True
                pick_time = 0
                while flag:
                    idx1 = random.choice(range(len(items)))
                    context_id1 = items[idx1]
                    candidates = items[idx1 + 10 : idx1 + 51]
                    candidates_stay = []
                    for idx2, candidate in enumerate(candidates):
                        if (
                            iou[context_id1, candidate] > 0.3
                            and iou[context_id1, candidate] < 0.8
                        ):
                            candidates_stay.append((idx2, candidate))
                    if (
                        len(candidates_stay)
                        <= self.num_extra_target_views + self.num_extra_context_views
                    ):
                        pick_time += 1
                        if pick_time > 100:
                            raise ValueError(
                                f"Cannot find enough target views for context view {context_id1} in scan {scan_name}"
                            )
                        continue
                    idx2, context_id2 = random.choice(candidates_stay)
                    context_views_id = [context_id1, context_id2]
                    extra_views_id = random.sample(
                        items[idx1 + 1 : idx1 + idx2 + 10],
                        self.num_extra_context_views + self.num_extra_target_views,
                    )
                    # divide the extra views into context and target
                    extra_context_views_id = extra_views_id[
                        : self.num_extra_context_views
                    ]
                    extra_target_views_id = extra_views_id[
                        self.num_extra_context_views : self.num_extra_context_views
                        + self.num_extra_target_views
                    ]
                    context_views_id = sorted(context_views_id + extra_context_views_id)
                    target_views_id = sorted(context_views_id + extra_target_views_id)
                    break
            else:
                pairs = self.val_pairs[idx]
                scan_name = pairs["scan"]
                scan_path = osp.join(self.scans_dir, scan_name)
                context_views_id = pairs["context_ids"]
                target_views_id = pairs["target_ids"]

            context_views_images = [
                np.array(
                    Image.open(osp.join(scan_path, "color", f"{context_view}.png"))
                )
                for context_view in context_views_id
            ]
            target_views_images = [
                np.array(Image.open(osp.join(scan_path, "color", f"{target_view}.png")))
                for target_view in target_views_id
            ]
            context_views_depths = [
                np.array(
                    Image.open(osp.join(scan_path, "depth", f"{context_view}.png"))
                )
                / 1000.0
                for context_view in context_views_id
            ]
            target_views_depths = [
                np.array(Image.open(osp.join(scan_path, "depth", f"{target_view}.png")))
                / 1000.0
                for target_view in target_views_id
            ]
            intrinsic = np.loadtxt(osp.join(scan_path, "intrinsic.txt"))
            context_views_intrinsics = [intrinsic for context_view in context_views_id]
            context_views_extrinsics = [
                np.loadtxt(osp.join(scan_path, "extrinsic", f"{context_view}.txt"))
                for context_view in context_views_id
            ]
            target_views_intrinsics = [intrinsic for target_view in target_views_id]
            target_views_extrinsics = [
                np.loadtxt(osp.join(scan_path, "extrinsic", f"{target_view}.txt"))
                for target_view in target_views_id
            ]
            context_views_extrinsics, target_views_extrinsics = self.relative_pose(
                context_views_extrinsics, target_views_extrinsics
            )
            context_views_intrinsics = self.intrinsics_normalize(
                context_views_intrinsics
            )
            target_views_intrinsics = self.intrinsics_normalize(target_views_intrinsics)
            # (H, W, 3) -> (3, H, W)
            context_views_images = [
                np.transpose(context_view_image, (2, 0, 1))
                for context_view_image in context_views_images
            ]
            target_views_images = [
                np.transpose(target_view_image, (2, 0, 1))
                for target_view_image in target_views_images
            ]

            if self.seg_task == "panoptic":
                context_views_segm = [
                    np.array(
                        Image.open(
                            osp.join(scan_path, "panoptic", f"{context_view}.png")
                        )
                    )
                    for context_view in context_views_id
                ]
                target_views_segm = [
                    np.array(
                        Image.open(
                            osp.join(scan_path, "panoptic", f"{target_view}.png")
                        )
                    )
                    for target_view in target_views_id
                ]
            elif self.seg_task == "instance":
                context_views_segm = [
                    np.array(
                        Image.open(
                            osp.join(scan_path, "instance", f"{context_view}.png")
                        )
                    )
                    for context_view in context_views_id
                ]
                target_views_segm = [
                    np.array(
                        Image.open(
                            osp.join(scan_path, "instance", f"{target_view}.png")
                        )
                    )
                    for target_view in target_views_id
                ]
            else:
                raise ValueError(f"Unknown segmentation task {self.seg_task}")
            context_views_segm = [
                context_view_segm[:, :, 0]
                + context_view_segm[:, :, 1] * 256
                + context_view_segm[:, :, 2] * 256 * 256
                for context_view_segm in context_views_segm
            ]
            context_views_semantic = [
                context_view_segm // 1000 for context_view_segm in context_views_segm
            ]
            context_views_instance = [
                context_view_segm % 1000 for context_view_segm in context_views_segm
            ]
            context_views_semantic_labels = [
                np.unique(context_view_semantic)
                for context_view_semantic in context_views_semantic
            ]
            segment_id2semantic_id = []
            for ind, semantic_labels in enumerate(context_views_semantic_labels):
                if len(semantic_labels) == 1 and semantic_labels[0] == 0:
                    raise ValueError(
                        f"No semantic label in the scene {scan_name} context view {context_views_id[ind]}"
                    )
                ins2sem = {}
                for semantic_label in semantic_labels:
                    segment_ids = np.unique(
                        context_views_instance[ind][
                            context_views_semantic[ind] == semantic_label
                        ]
                    )
                    ins2sem.update(
                        {segment_id: semantic_label for segment_id in segment_ids}
                    )
                segment_id2semantic_id.append(ins2sem)
            mask2former_inputs_context = self.processor.preprocess(
                video_frames=context_views_images,
                segmentation_maps=context_views_instance,
                instance_id_to_semantic_id=segment_id2semantic_id,
                return_tensors="pt",
            )
            target_views_segm = [
                target_view_segm[:, :, 0]
                + target_view_segm[:, :, 1] * 256
                + target_view_segm[:, :, 2] * 256 * 256
                for target_view_segm in target_views_segm
            ]
            target_views_semantic = [
                target_view_segm // 1000 for target_view_segm in target_views_segm
            ]
            target_views_instance = [
                target_view_segm % 1000 for target_view_segm in target_views_segm
            ]
            target_views_semantic_labels = [
                np.unique(target_view_semantic)
                for target_view_semantic in target_views_semantic
            ]
            segment_id2semantic_id = []
            for ind, semantic_labels in enumerate(target_views_semantic_labels):
                if len(semantic_labels) == 1 and semantic_labels[0] == 0:
                    raise ValueError(
                        f"No semantic label in the scene {scan_name} target view {target_views_id[ind]}"
                    )
                ins2sem = {}
                for semantic_label in semantic_labels:
                    segment_ids = np.unique(
                        target_views_instance[ind][
                            target_views_semantic[ind] == semantic_label
                        ]
                    )
                    ins2sem.update(
                        {segment_id: semantic_label for segment_id in segment_ids}
                    )
                segment_id2semantic_id.append(ins2sem)
            target_views_semantic_labels = [
                np.unique(target_view_semantic)
                for target_view_semantic in target_views_semantic
            ]
            mask2former_inputs_target = self.processor.preprocess(
                video_frames=target_views_images,
                segmentation_maps=target_views_instance,
                instance_id_to_semantic_id=segment_id2semantic_id,
                return_tensors="pt",
            )
            data = {
                "scene_names": scan_name,
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
                "context_mask_labels": mask2former_inputs_context.mask_labels,
                "context_class_labels": mask2former_inputs_context.class_labels,
                "target_mask_labels": mask2former_inputs_target.mask_labels,
                "target_class_labels": mask2former_inputs_target.class_labels,
            }
            return data
        except Exception as e:
            if isinstance(e, ValueError):
                log.warning(e)
                if self.train:
                    return self.__getitem__((idx + 1) % self.__len__())
                else:
                    return self.__getitem__((idx + 1) % self.__len__())
            else:
                raise e
