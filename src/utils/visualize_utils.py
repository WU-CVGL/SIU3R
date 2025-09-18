import torch
import numpy as np
from PIL import Image
from typing import Optional
from torch import Tensor
import cv2
import kornia
from pathlib import Path
import os
import json
import plyfile
from copy import deepcopy
from src.utils.scannet_constant import (
    PANOPTIC_COLOR_PALLETE,
    INSTANCE_COLOR_PALLETE,
    PANOPTIC_SEMANTIC2NAME,
    PANOPTIC_CONTINUOUS2SEMANTIC,
)
from src.utils.ply_export import export_ply
from src.utils.gaussians_types import Gaussians
from torchvision.utils import save_image
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])


def save_sem_ids(
    sem_ids: list[Tensor],  # [(N, H, W)] * B
    target_masks_labels: list[Tensor],  # [(N, H, W)] * B
    target_classes_labels: list[Tensor],  # [(N)] * B
    save_dir: str,
    scene_names: list[str],  # [B]
    context_views_id: list[int],  # [B]
    target_views_id: list[int],  # [B*[N]]
    global_rank: int,
    COLOR_PALLETE: dict[int, list[int]] = PANOPTIC_COLOR_PALLETE,
):
    N, H, W = sem_ids[0].shape
    for (
        sem_id,
        mask_label,
        class_label,
        scene_name,
        context_view_id,
        target_view_id,
    ) in zip(
        sem_ids,
        target_masks_labels,
        target_classes_labels,
        scene_names,
        context_views_id,
        target_views_id,
    ):
        context_view_str = "_".join(map(str, context_view_id))
        base_path = f"{save_dir}/{scene_name}_context{context_view_str}"
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(f"{base_path}/target_seg_sem_pred", exist_ok=True)
        os.makedirs(f"{base_path}/target_seg_sem_pred_color", exist_ok=True)
        os.makedirs(f"{base_path}/target_seg_sem_gt", exist_ok=True)
        os.makedirs(f"{base_path}/target_seg_sem_gt_color", exist_ok=True)
        for sem_id, vid in zip(sem_id, target_view_id):
            sem_map = sem_id.cpu().numpy().astype(np.uint8)
            color_sem_map = np.zeros((H, W, 3), dtype=np.uint8)
            for sem_id in np.unique(sem_map):
                if sem_id == 0:
                    continue
                color_sem_map[sem_map == sem_id] = COLOR_PALLETE[sem_id]
            path = Path(f"{base_path}/target_seg_sem_pred/{scene_name}_pred{vid}.png")
            color_path = Path(
                f"{base_path}/target_seg_sem_pred_color/{scene_name}_pred_{vid}.png"
            )
            sem_map = Image.fromarray(sem_map)
            sem_map.save(path)
            color_sem_map = Image.fromarray(color_sem_map)
            color_sem_map.save(color_path)
        gt_semantic_id = torch.zeros(N, H, W, dtype=torch.int64)
        for ins_id, (mask, cls) in enumerate(zip(mask_label, class_label)):
            gt_semantic_id[mask == 1] = cls + 1
        for gt_sem, vid in zip(gt_semantic_id, target_view_id):
            gt_sem_map = gt_sem.cpu().numpy().astype(np.uint8)
            color_gt_sem_map = np.zeros((H, W, 3), dtype=np.uint8)
            for sem_id in np.unique(gt_sem_map):
                if sem_id == 0:
                    continue
                color_gt_sem_map[gt_sem_map == sem_id] = COLOR_PALLETE[sem_id]
            path = Path(f"{base_path}/target_seg_sem_gt/{scene_name}_gt{vid}.png")
            color_path = Path(
                f"{base_path}/target_seg_sem_gt_color/{scene_name}_gt_{vid}.png"
            )
            gt_sem_map = Image.fromarray(gt_sem_map)
            gt_sem_map.save(path)
            color_gt_sem_map = Image.fromarray(color_gt_sem_map)
            color_gt_sem_map.save(color_path)


def save_id_maps(
    sem_ids: list[Tensor],  # [(N, H, W)] * B
    ins_ids: list[Tensor],  # [(N, H, W)] * B
    target_masks_labels: list[Tensor],  # [(N, H, W)] * B
    target_classes_labels: list[Tensor],  # [(N)] * B
    save_dir: str,
    scene_names: list[str],  # [B]
    context_views_id: list[int],  # [B]
    target_views_id: list[int],  # [B*[N]]
    global_rank: int,
):
    N, H, W = sem_ids[0].shape
    for (
        sem_id,
        ins_id,
        mask_label,
        class_label,
        scene_name,
        context_view_id,
        target_view_id,
    ) in zip(
        sem_ids,
        ins_ids,
        target_masks_labels,
        target_classes_labels,
        scene_names,
        context_views_id,
        target_views_id,
    ):
        context_view_str = "_".join(map(str, context_view_id))
        base_path = f"{save_dir}/{scene_name}_context{context_view_str}"
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(f"{base_path}/target_seg_pred", exist_ok=True)
        os.makedirs(f"{base_path}/target_seg_gt", exist_ok=True)
        segment_id = 1000 * sem_id + ins_id
        for seg_id, vid in zip(segment_id, target_view_id):
            segment_map = torch.zeros(3, H, W, dtype=torch.uint8)
            segment_map[0, ...] = seg_id % 256
            segment_map[1, ...] = seg_id // 256
            segment_map[2, ...] = seg_id // 256 // 256
            segment_map = segment_map.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            path = Path(f"{base_path}/target_seg_pred/{scene_name}_pred{vid}.png")
            segment_map = Image.fromarray(segment_map)
            segment_map.save(path)
        gt_instance_id = torch.zeros(N, H, W, dtype=torch.int64)
        gt_semantic_id = torch.zeros(N, H, W, dtype=torch.int64)
        for ins_id, (mask, cls) in enumerate(zip(mask_label, class_label)):
            gt_instance_id[mask == 1] = ins_id + 1
            gt_semantic_id[mask == 1] = cls + 1
        gt_segment_id = 1000 * gt_semantic_id + gt_instance_id
        for gt_seg, vid in zip(gt_segment_id, target_view_id):
            gt_segment_map = torch.zeros(3, H, W, dtype=torch.uint8)
            gt_segment_map[0, ...] = gt_seg % 256
            gt_segment_map[1, ...] = gt_seg // 256
            gt_segment_map[2, ...] = gt_seg // 256 // 256
            gt_segment_map = (
                gt_segment_map.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            )
            path = Path(f"{base_path}/target_seg_gt/{scene_name}_gt{vid}.png")
            gt_segment_map = Image.fromarray(gt_segment_map)
            gt_segment_map.save(path)


def save_seg_masks(
    seg_masks: list[Tensor],
    seg_info: list[list[dict]],
    mask_labels: list[Tensor],
    class_labels: list[Tensor],
    save_dir: str,
    scene_names: list[str],  # [B]
    context_views_id: list[list[int]],  # [B]
    target_views_id: list[list[int]],  # [B*[N]]
    global_rank: int,
    target_seg_masks: list[Tensor] = None,
    target_seg_infos: list[list[dict]] = None,
    target_masks_labels: list[Tensor] = None,
    target_classes_labels: list[Tensor] = None,
):
    for idx, (
        seg_mask,
        seg,
        mask_label,
        class_label,
        scene_name,
        context_view_id,
    ) in enumerate(
        zip(
            seg_masks,
            seg_info,
            mask_labels,
            class_labels,
            scene_names,
            context_views_id,
        )
    ):
        context_view_str = "_".join(map(str, context_views_id[idx]))
        base_path = f"{save_dir}/{scene_name}_context{context_view_str}"
        os.makedirs(base_path, exist_ok=True)
        if os.path.exists(f"{base_path}/context_seg_pred/pred.json"):
            continue
        os.makedirs(f"{base_path}/context_seg_pred", exist_ok=True)
        os.makedirs(f"{base_path}/context_seg_gt", exist_ok=True)
        N, H, W = seg_mask.shape
        instance_id = seg_mask.clone()
        semantic_id = torch.zeros_like(instance_id)
        for info in seg:
            semantic_id[seg_mask == info["id"]] = info["label_id"] + 1
        seg_save = deepcopy(seg)
        for info in seg_save:
            info["label_id"] += 1
        json.dump(
            seg_save, open(f"{base_path}/context_seg_pred/pred.json", "w"), indent=4
        )
        instance_id[semantic_id == 0] = 0
        segment_id = 1000 * semantic_id + instance_id
        for seg_id, vid in zip(segment_id, context_view_id):
            segment_map = torch.zeros(3, H, W, dtype=torch.uint8)
            segment_map[0, ...] = seg_id % 256
            segment_map[1, ...] = seg_id // 256
            segment_map[2, ...] = seg_id // 256 // 256
            segment_map = segment_map.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            path = Path(f"{base_path}/context_seg_pred/{scene_name}_pred{vid}.png")
            segment_map = Image.fromarray(segment_map)
            segment_map.save(path)
        gt_instance_id = torch.zeros(N, H, W, dtype=torch.int64)
        gt_semantic_id = torch.zeros(N, H, W, dtype=torch.int64)
        for ins_id, (mask, cls) in enumerate(zip(mask_label, class_label)):
            gt_instance_id[mask == 1] = ins_id + 1
            gt_semantic_id[mask == 1] = cls + 1
        gt_segment_id = 1000 * gt_semantic_id + gt_instance_id
        for gt_seg, vid in zip(gt_segment_id, context_view_id):
            gt_segment_map = torch.zeros(3, H, W, dtype=torch.uint8)
            gt_segment_map[0, ...] = gt_seg % 256
            gt_segment_map[1, ...] = gt_seg // 256
            gt_segment_map[2, ...] = gt_seg // 256 // 256
            gt_segment_map = (
                gt_segment_map.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            )
            path = Path(f"{base_path}/context_seg_gt/{scene_name}_gt{vid}.png")
            gt_segment_map = Image.fromarray(gt_segment_map)
            gt_segment_map.save(path)
    if target_seg_masks is not None and target_seg_infos is not None:
        for idx, (
            seg_mask,
            seg,
            mask_label,
            class_label,
            scene_name,
            context_view_id,
            target_view_id,
        ) in enumerate(
            zip(
                target_seg_masks,
                target_seg_infos,
                target_masks_labels,
                target_classes_labels,
                scene_names,
                context_views_id,
                target_views_id,
            )
        ):
            context_view_str = "_".join(map(str, context_views_id[idx]))
            base_path = f"{save_dir}/{scene_name}_context{context_view_str}"
            os.makedirs(base_path, exist_ok=True)
            if os.path.exists(f"{base_path}/target_seg_pred/pred.json"):
                continue
            os.makedirs(f"{base_path}/target_seg_pred", exist_ok=True)
            os.makedirs(f"{base_path}/target_seg_gt", exist_ok=True)
            N, H, W = seg_mask.shape
            instance_id = seg_mask.clone()
            semantic_id = torch.zeros_like(instance_id)
            for info in seg:
                semantic_id[seg_mask == info["id"]] = info["label_id"] + 1
            seg_save = deepcopy(seg)
            for info in seg_save:
                info["label_id"] += 1
            json.dump(
                seg_save, open(f"{base_path}/target_seg_pred/pred.json", "w"), indent=4
            )
            instance_id[semantic_id == 0] = 0
            segment_id = 1000 * semantic_id + instance_id
            for seg_id, vid in zip(segment_id, target_view_id):
                segment_map = torch.zeros(3, H, W, dtype=torch.uint8)
                segment_map[0, ...] = seg_id % 256
                segment_map[1, ...] = seg_id // 256
                segment_map[2, ...] = seg_id // 256 // 256
                segment_map = (
                    segment_map.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                )
                path = Path(f"{base_path}/target_seg_pred/{scene_name}_pred{vid}.png")
                segment_map = Image.fromarray(segment_map)
                segment_map.save(path)
            gt_instance_id = torch.zeros(N, H, W, dtype=torch.int64)
            gt_semantic_id = torch.zeros(N, H, W, dtype=torch.int64)
            for ins_id, (mask, cls) in enumerate(zip(mask_label, class_label)):
                gt_instance_id[mask == 1] = ins_id + 1
                gt_semantic_id[mask == 1] = cls + 1
            gt_segment_id = 1000 * gt_semantic_id + gt_instance_id
            for gt_seg, vid in zip(gt_segment_id, target_view_id):
                gt_segment_map = torch.zeros(3, H, W, dtype=torch.uint8)
                gt_segment_map[0, ...] = gt_seg % 256
                gt_segment_map[1, ...] = gt_seg // 256
                gt_segment_map[2, ...] = gt_seg // 256 // 256
                gt_segment_map = (
                    gt_segment_map.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                )
                path = Path(f"{base_path}/target_seg_gt/{scene_name}_gt{vid}.png")
                gt_segment_map = Image.fromarray(gt_segment_map)
                gt_segment_map.save(path)








def visualize_seg_mask(
    images: Tensor,  # (B, 2, C, H, W)
    masks: list[Tensor],  # [(2, H, W)] * B
    segment_infos: list[dict],  # [{"id": int, "label_id": int, "score": float}*N] * B
    mask_labels: Optional[list[Tensor]],  # [(N, 2, H, W)] * B
    class_labels: Optional[list[Tensor]],  # [(N)] * B
    save_dir: str,
    scene_names: list[str],
    context_views: list[list[int]],
    target_views: list[list[int]],
    global_rank: int,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    SEMANTIC2NAME: dict[int, str] = PANOPTIC_SEMANTIC2NAME,
    CONTINUOUS2SEMANTIC: dict[int, int] = PANOPTIC_CONTINUOUS2SEMANTIC,
    COLOR_PALLETE: dict[int, list[int]] = PANOPTIC_COLOR_PALLETE,
    ramdom_color: bool = False,
    alpha=0.5,
):
    try:
        B, N, C, H, W = images.shape
        images = [
            unnormalize_image(img, mean, std) for img in images
        ]  # [(2, H, W, C)] * B
        masks = [m.cpu().numpy() for m in masks]  # [(2, H, W)] * B
        overlayed_images = []
        for i, (image, mask, segment_info) in enumerate(
            zip(images, masks, segment_infos)
        ):
            # image: (2, H, W, C), mask: (2, H, W), segment_info: [{"id": int, "label_id": int, "score": float}]*num_instances
            colored_masks = []
            for im, mas in zip(image, mask):
                # im: (H, W, C), mas: (H, W)
                colored_mask = np.zeros_like(im, dtype=np.uint8)  # (H, W, C)
                color = None
                for seg_info in segment_info:
                    semantic_id = CONTINUOUS2SEMANTIC[seg_info["label_id"] + 1]
                    if semantic_id == 0:
                        continue
                    if not ramdom_color:
                        color = COLOR_PALLETE[semantic_id]
                    else:
                        color = np.random.randint(0, 255, 3)
                    # draw mask
                    colored_mask[mas == seg_info["id"]] = color  # (H, W, C)
                for seg_info in segment_info:
                    semantic_id = CONTINUOUS2SEMANTIC[seg_info["label_id"] + 1]
                    if semantic_id == 0:
                        continue
                    if not ramdom_color:
                        color = COLOR_PALLETE[semantic_id]
                    else:
                        color = np.random.randint(0, 255, 3)
                    mask_area = mas == seg_info["id"]  # (H, W)
                    # Find contours for the mask
                    contours, _ = cv2.findContours(
                        (mask_area).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    if len(contours) == 0:
                        continue
                    x_min, y_min = float("inf"), float("inf")
                    x_max, y_max = -float("inf"), -float("inf")
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x + w)
                        y_max = max(y_max, y + h)
                    colored_mask = np.ascontiguousarray(colored_mask)
                    cv2.rectangle(
                        colored_mask,  # (H, W, C)
                        (x_min, y_min),
                        (x_max, y_max),
                        color,
                        2,
                    )
                    cv2.drawContours(colored_mask, contours, -1, (255, 255, 255), 2)
                    # Draw category text in the center of the bounding box
                    category_text = f"{seg_info['id']}|{SEMANTIC2NAME[semantic_id]}|{seg_info['score']:.2f}"
                    text_size = cv2.getTextSize(
                        category_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )[0]
                    text_x = x_min + (x_max - x_min - text_size[0]) // 2
                    text_y = y_min + (y_max - y_min + text_size[1]) // 2
                    # draw a rectangle for the text
                    cv2.rectangle(
                        colored_mask,
                        (text_x - 3, text_y - text_size[1] - 2),
                        (text_x + text_size[0] + 3, text_y + 2),
                        color,
                        -1,
                    )
                    cv2.putText(
                        colored_mask,
                        category_text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )  # Black text
                colored_masks.append(colored_mask)
            colored_masks = np.concatenate(colored_masks, axis=1)  # (H, 2*W, C)
            # only overlay mask not background
            overlayed_image = image.transpose(1, 0, 2, 3).reshape(
                H, N * W, C
            )  # (H, 2*W, C)
            overlayed_image[colored_masks != 0] = (
                alpha * colored_masks[colored_masks != 0]
                + (1 - alpha) * overlayed_image[colored_masks != 0]
            )
            overlayed_images.append(overlayed_image)
        batch_gt_images = []
        if mask_labels is not None and class_labels is not None:
            mask_labels = [m.cpu().numpy() for m in mask_labels]  # [(N, 2, H, W)] * B
            class_labels = [c.cpu().numpy() for c in class_labels]  # [(N)] * B
            for i, (image, mask_label, class_label) in enumerate(
                zip(images, mask_labels, class_labels)
            ):
                gt_images = []
                # image: (2, H, W, C), mask_label: (N, 2, H, W), class_label: (N)
                mask_label = mask_label.transpose(1, 0, 2, 3)  # (2, N, H, W)
                for im, mas in zip(image, mask_label):
                    # im: (H, W, C), mas: (N, H, W)
                    gt_image = np.zeros_like(im, dtype=np.uint8)  # (H, W, C)
                    for mask, cls in zip(mas, class_label):  # iteration over N
                        mask_areas = mask == 1  # (H, W)
                        semantic_id = CONTINUOUS2SEMANTIC[cls + 1]
                        if semantic_id == 0:
                            continue
                        color = COLOR_PALLETE[semantic_id]
                        gt_image[mask_areas] = color
                        # draw bounding box
                        contours, _ = cv2.findContours(
                            (mask_areas).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        if len(contours) == 0:
                            continue
                        x_min, y_min = float("inf"), float("inf")
                        x_max, y_max = -float("inf"), -float("inf")
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            x_min = min(x_min, x)
                            y_min = min(y_min, y)
                            x_max = max(x_max, x + w)
                            y_max = max(y_max, y + h)
                        gt_image = np.ascontiguousarray(gt_image)
                        cv2.rectangle(
                            img=gt_image,
                            pt1=(x_min, y_min),
                            pt2=(x_max, y_max),
                            color=color,
                            thickness=2,
                        )
                        cv2.drawContours(gt_image, contours, -1, (255, 255, 255), 2)
                        category_text = f"{SEMANTIC2NAME[semantic_id]}"
                        text_size = cv2.getTextSize(
                            category_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )[0]
                        text_x = x_min + (x_max - x_min - text_size[0]) // 2
                        text_y = y_min + (y_max - y_min + text_size[1]) // 2
                        cv2.rectangle(
                            gt_image,
                            (text_x - 3, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 3, text_y + 2),
                            color,
                            -1,
                        )
                        cv2.putText(
                            gt_image,
                            category_text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    gt_images.append(gt_image)
                gt_images = np.concatenate(gt_images, axis=1)  # (H, 2*W, C)
                batch_gt_images.append(gt_images)
        for i in range(B):
            scene_name = scene_names[i]
            context_views_str = "_".join(map(str, context_views[i]))
            if target_views is not None:
                target_views_str = "_".join(map(str, target_views[i]))
            else:
                target_views_str = context_views_str
            os.makedirs(
                f"{save_dir}/{scene_name}_context{context_views_str}", exist_ok=True
            )
            path = f"{save_dir}/{scene_name}_context{context_views_str}/{scene_name}_seg{target_views_str}.png"
            if os.path.exists(path):
                continue
            overlayed_image = overlayed_images[i]  # (H, 2*W, C)
            if mask_labels is not None and class_labels is not None:
                gt_image = batch_gt_images[i]  # (H, 2*W, C)
                overlayed_image = np.concatenate(
                    [overlayed_image, gt_image], axis=0
                )  # (2*H, 2*W, C)
            overlayed_image = Image.fromarray(overlayed_image)
            overlayed_image.save(path)
    except Exception as e:
        log.error(f"Error in visualize_seg_mask: {e}")
        raise e


def unnormalize_image(
    image: Tensor,  # (N, C, H, W)
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    image = image.cpu().numpy().transpose(0, 2, 3, 1)
    if mean is not None and std is not None:
        image = image * std + mean
    image = (image * 255).astype(np.uint8)
    return image


def visualize_semantic_map(
    semantic_map: Tensor,  # (H, W)
    save_dir: str,
    scene_name: str,
    context_view: str,
    SEMANTIC2NAME: dict[int, str] = PANOPTIC_SEMANTIC2NAME,
    COLOR_PALLETE: dict[int, list[int]] = PANOPTIC_COLOR_PALLETE,
):
    semantic_map = semantic_map.cpu().numpy()
    H, W = semantic_map.shape
    colored_map = np.zeros((H, W, 3), dtype=np.uint8)
    for label_id in SEMANTIC2NAME.keys():
        colored_map[semantic_map == label_id] = COLOR_PALLETE[label_id]
    path = f"{save_dir}/semantic_{scene_name}_{context_view}.png"
    colored_map = Image.fromarray(colored_map)
    colored_map.save(path)


def visualize_instance_map(
    instance_map: Tensor,  # (H, W)
    save_dir: str,
    scene_name: str,
    context_view: str,
):
    instance_map = instance_map.cpu().numpy()
    H, W = instance_map.shape
    colored_map = np.zeros((H, W, 3), dtype=np.uint8)
    instance_ids = np.unique(instance_map)
    for instance_id in instance_ids:
        if instance_id == 0:
            continue
        colored_map[instance_map == instance_id] = np.random.randint(0, 255, 3)
    path = f"{save_dir}/instance_{scene_name}_{context_view}.png"
    colored_map = Image.fromarray(colored_map)
    colored_map.save(path)


def visualize_mask_labels(
    mask_labels: list[Tensor],  # [(N, H, W)] * B
    class_labels: list[Tensor],  # [(N)] * B
    save_dir: str,
    scene_name: list[str],
    context_view: list[str],
    CONTINUOUS2SEMANTIC: dict[int, int] = PANOPTIC_CONTINUOUS2SEMANTIC,
    COLOR_PALLETE: dict[int, list[int]] = PANOPTIC_COLOR_PALLETE,
):
    # mask labels are binary masks
    mask_labels = [m.cpu().numpy() for m in mask_labels]
    class_labels = [c.cpu().numpy() for c in class_labels]
    for view, name, mask_label, class_label in zip(
        context_view, scene_name, mask_labels, class_labels
    ):
        result = np.zeros((mask_label.shape[1], mask_label.shape[2], 3), dtype=np.uint8)
        for mask_index, (mask, cls) in enumerate(zip(mask_label, class_label)):
            result[mask == 1] = COLOR_PALLETE[CONTINUOUS2SEMANTIC[cls]]
        path = f"{save_dir}/{name}_{view}.png"
        cv2.imwrite(path, result)
