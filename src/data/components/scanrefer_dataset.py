import os
import os.path as osp
import json
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.models.mask2former import VideoMask2FormerImageProcessor
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ScanReferDataset(Dataset):

    def __init__(
        self,
        root: str,
        num_extra_target_views: int = 2,
        image_width: int = 256,
        image_height: int = 256,
        train: bool = True,
        seg_task: str = "panoptic",
        val_pair_json: str = "val_pair.json",
    ):
        super().__init__()
        self.root = root
        self.train = train
        if self.train:
            self.scans_dir = osp.join(self.root, "train")
            with open(osp.join(self.root, "train_refer_seg_data.json"), "r") as f:
                self.refer_data = json.load(f)
        else:
            self.scans_dir = osp.join(self.root, "val")
            with open(osp.join(self.root, "val_refer_seg_data.json"), "r") as f:
                self.refer_data = json.load(f)
            with open(osp.join(self.root, "val_refer_pair.json"), "r") as f:
                self.val_pairs = json.load(f)
        scan_names = self.refer_data.keys()
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

    def __getitem__(self, idx: int):
        if self.train:
            scene_name = self.scan_names[idx]
            scan_path = osp.join(self.scans_dir, scene_name)
            data = self.refer_data[scene_name]
            frames = list(data["frame2object"].keys())
            frames = [int(frame) for frame in frames]
            frames = sorted(frames)
            # randomly pick a frame id, and get its idx in list
            choice_right_margin = len(frames) - 1 - 30
            if choice_right_margin <= 0:
                choice_right_margin = len(frames) - 1
            choice_frame_idx = random.randint(0, choice_right_margin)
            # randomly pick a range between 10 and 30 frames
            choice_range = random.randint(10, 30)
            choice_right_idx = choice_frame_idx + choice_range
            if choice_right_idx >= len(frames):
                choice_right_idx = len(frames) - 1
            context_views_id = [
                frames[choice_frame_idx],
                frames[choice_right_idx],
            ]
            context_objects = set(
                data["frame2object"][str(context_views_id[0])]
                + data["frame2object"][str(context_views_id[1])]
            )
            context_objects = sorted([int(obj_id) for obj_id in context_objects])
        else:
            scene_name = self.val_pairs[idx]["scene_name"]
            scan_path = osp.join(self.scans_dir, scene_name)
            data = self.refer_data[scene_name]
            context_views_id = self.val_pairs[idx]["context_views_id"]
            context_objects = [self.val_pairs[idx]["context_objects"]]

        context_views_images = [
            np.array(Image.open(osp.join(scan_path, "color", f"{context_view}.jpg")))
            for context_view in context_views_id
        ]
        context_views_images = [
            np.transpose(context_view_image, (2, 0, 1))
            for context_view_image in context_views_images
        ]
        intrinsic = np.loadtxt(osp.join(scan_path, "intrinsic.txt"))
        context_views_intrinsics = [intrinsic for context_view in context_views_id]
        context_views_intrinsics = self.intrinsics_normalize(context_views_intrinsics)
        context_views_segm = [
            np.array(Image.open(osp.join(scan_path, "panoptic", f"{context_view}.png")))
            for context_view in context_views_id
        ]
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
        # stack context_views_instance and turn it to tensor
        context_views_instance = torch.stack(
            [
                torch.from_numpy(context_view_instance).long()
                for context_view_instance in context_views_instance
            ]
        )
        context_mask_labels = []
        context_class_labels = []
        texts = []
        text_tokens = []
        for obj_id in context_objects:
            obj = data["objects"][str(obj_id)]
            obj_panoptic_label_id = obj["panoptic_label_id"]
            obj_texts = obj["text"]
            obj_text_tokens = obj["text_token"]
            choice_text_idx = random.randint(0, len(obj_texts) - 1)
            choice_text = obj_texts[choice_text_idx]
            choice_text_token = torch.tensor(
                obj_text_tokens[choice_text_idx], dtype=torch.long
            )
            context_mask_labels.append(context_views_instance == obj_id)
            context_class_labels.append(obj_panoptic_label_id - 1)
            texts.append(choice_text)
            text_tokens.append(choice_text_token)
        context_mask_labels = torch.stack(context_mask_labels)
        context_class_labels = torch.tensor(context_class_labels, dtype=torch.long)
        text_tokens = torch.stack(text_tokens)
        data = {
            "context_views_images": context_views_images,
            "context_views_intrinsics": context_views_intrinsics,
            "context_mask_labels": context_mask_labels,
            "context_class_labels": context_class_labels,
            "text": texts,
            "text_token": text_tokens,
            "scene_names": scene_name,
            "context_views_id": context_views_id,
        }
        return data
