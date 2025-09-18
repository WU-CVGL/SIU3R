import os
import numpy as np
import torch
from torch import Tensor
from src.config import VisualizerCfg
from typing import Literal
import kornia
import cv2
from PIL import Image
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from src.utils.gaussians_types import Gaussians
from src.utils.ply_export import export_ply
from src.utils.scannet_constant import (
    PANOPTIC_SEMANTIC2NAME,
    PANOPTIC_CONTINUOUS2SEMANTIC,
    PANOPTIC_COLOR_PALLETE,
)
from src.utils.coco_constant import (
    ADE20K_PANOPTIC_SEMANTIC2NAME,
    ADE20K_PANOPTIC_CONTINUOUS2SEMANTIC,
    ADE20K_PANOPTIC_COLOR_PALLETE,
    COCO_PANOPTIC_SEMANTIC2NAME,
    COCO_PANOPTIC_CONTINUOUS2SEMANTIC,
    COCO_PANOPTIC_COLOR_PALLETE,
)

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class Visualizer:
    def __init__(self, cfg: VisualizerCfg):
        self.cfg = cfg
        self.pool = []
        self.semantic2name = None
        self.continuous2semantic = None
        self.color_palette = None

        self.semantic2name = PANOPTIC_SEMANTIC2NAME
        self.continuous2semantic = PANOPTIC_CONTINUOUS2SEMANTIC
        self.color_palette = PANOPTIC_COLOR_PALLETE

        # self.semantic2name = ADE20K_PANOPTIC_SEMANTIC2NAME
        # self.continuous2semantic = ADE20K_PANOPTIC_CONTINUOUS2SEMANTIC
        # self.color_palette = ADE20K_PANOPTIC_COLOR_PALLETE

        # self.semantic2name = COCO_PANOPTIC_SEMANTIC2NAME
        # self.continuous2semantic = COCO_PANOPTIC_CONTINUOUS2SEMANTIC
        # self.color_palette = COCO_PANOPTIC_COLOR_PALLETE

    def reset(self):
        self.pool = []
        torch.cuda.empty_cache()

    def add(
        self,
        save_dir: str,
        batch=None,
        render_output=None,
        gaussians=None,
        context_seg_masks=None,
        context_seg_infos=None,
        context_semantic_ids=None,
        context_instance_ids=None,
        target_seg_masks=None,
        target_seg_infos=None,
        target_semantic_ids=None,
        target_instance_ids=None,
    ):
        added_data = {}
        added_data["save_dir"] = save_dir
        if "scene_names" in batch:
            added_data["scene_names"] = batch["scene_names"]
        if "context_views_id" in batch:
            added_data["context_views_id"] = batch["context_views_id"]
        if "target_views_id" in batch:
            added_data["target_views_id"] = batch["target_views_id"]
        if "context_views_images" in batch:
            added_data["context_views_images"] = (
                batch["context_views_images"].detach().cpu()
            )
        if "context_views_depths" in batch:
            added_data["context_views_depths"] = (
                batch["context_views_depths"].detach().cpu()
            )
        if "target_views_images" in batch:
            added_data["target_views_images"] = (
                batch["target_views_images"].detach().cpu()
            )
        if "target_views_depths" in batch:
            added_data["target_views_depths"] = (
                batch["target_views_depths"].detach().cpu()
            )
        if render_output is not None and "render_color" in render_output:
            added_data["render_color"] = render_output["render_color"].detach().cpu()
        if render_output is not None and "render_depth" in render_output:
            added_data["render_depth"] = render_output["render_depth"].detach().cpu()
        if "context_mask_labels" in batch:
            added_data["context_mask_labels"] = [
                mask.detach().cpu() for mask in batch["context_mask_labels"]
            ]
        if "context_class_labels" in batch:
            added_data["context_class_labels"] = [
                label.detach().cpu() for label in batch["context_class_labels"]
            ]
        if "target_mask_labels" in batch:
            added_data["target_mask_labels"] = [
                mask.detach().cpu() for mask in batch["target_mask_labels"]
            ]
        if "target_class_labels" in batch:
            added_data["target_class_labels"] = [
                label.detach().cpu() for label in batch["target_class_labels"]
            ]

        if self.cfg.log_gaussian_ply:
            added_data["gaussians"] = gaussians.detach_cpu_copy()

        if target_instance_ids is not None and target_semantic_ids is not None:
            added_data["context_semantic_ids"] = context_semantic_ids.detach().cpu()
            added_data["context_instance_ids"] = context_instance_ids.detach().cpu()
            added_data["target_semantic_ids"] = target_semantic_ids.detach().cpu()
            added_data["target_instance_ids"] = target_instance_ids.detach().cpu()
            added_data["context_seg_infos"] = context_seg_infos
            added_data["target_seg_infos"] = target_seg_infos
        elif context_seg_masks is not None and context_seg_infos is not None:
            added_data["context_seg_masks"] = [
                mask.detach().cpu() for mask in context_seg_masks
            ]
            added_data["context_seg_infos"] = context_seg_infos
        self.pool.append(added_data)

    def write_file(self, data):
        scene_names = data["scene_names"]
        context_views_id = (
            data["context_views_id"] if "context_views_id" in data else None
        )
        target_views_id = data["target_views_id"] if "target_views_id" in data else None
        context_views_images = (
            data["context_views_images"] if "context_views_images" in data else None
        )
        context_views_depths = (
            data["context_views_depths"] if "context_views_depths" in data else None
        )
        target_views_images = (
            data["target_views_images"] if "target_views_images" in data else None
        )
        target_views_depths = (
            data["target_views_depths"] if "target_views_depths" in data else None
        )
        render_color = data["render_color"] if "render_color" in data else None
        render_depth = data["render_depth"] if "render_depth" in data else None
        save_dir = data["save_dir"]
        if render_color is not None and render_depth is not None:
            self.visualize_recon_image(
                rendered_images=render_color,
                rendered_depths=render_depth,
                gt_images=target_views_images,
                gt_depths=target_views_depths,
                save_dir=save_dir,
                scene_names=scene_names,
                context_views_id=context_views_id,
                target_views_id=target_views_id,
            )
        if self.cfg.log_gaussian_ply:
            self.save_gaussians(
                gaussians=data["gaussians"],
                save_dir=save_dir,
                scene_names=scene_names,
                context_views=context_views_id,
            )
        if "context_semantic_ids" in data and "context_instance_ids" in data:
            self.save_seg_ids(
                mode="context",
                semantic_ids=data["context_semantic_ids"],
                instance_ids=data["context_instance_ids"],
                save_dir=save_dir,
                scene_names=scene_names,
                context_views_id=context_views_id,
                seg_infos=data["context_seg_infos"],
            )
            self.save_gt_seg_masks(
                mode="context",
                mask_labels=data["context_mask_labels"],
                class_labels=data["context_class_labels"],
                save_dir=save_dir,
                scene_names=scene_names,
                context_views_id=context_views_id,
            )
            self.draw_overlay_segm_ids(
                gt_images=context_views_images,
                semantic_ids=data["context_semantic_ids"],
                instance_ids=data["context_instance_ids"],
                mask_labels=data["context_mask_labels"],
                class_labels=data["context_class_labels"],
                save_dir=save_dir,
                scene_names=scene_names,
                context_views_id=context_views_id,
            )
            self.save_seg_ids(
                mode="target",
                semantic_ids=data["target_semantic_ids"],
                instance_ids=data["target_instance_ids"],
                save_dir=save_dir,
                scene_names=scene_names,
                context_views_id=context_views_id,
                target_views_id=target_views_id,
                seg_infos=data["target_seg_infos"],
            )
            self.save_gt_seg_masks(
                mode="target",
                mask_labels=data["target_mask_labels"],
                class_labels=data["target_class_labels"],
                save_dir=save_dir,
                scene_names=scene_names,
                context_views_id=context_views_id,
                target_views_id=target_views_id,
            )
            self.draw_overlay_segm_ids(
                gt_images=target_views_images,
                semantic_ids=data["target_semantic_ids"],
                instance_ids=data["target_instance_ids"],
                mask_labels=data["target_mask_labels"],
                class_labels=data["target_class_labels"],
                save_dir=save_dir,
                scene_names=scene_names,
                context_views_id=context_views_id,
                target_views_id=target_views_id,
            )
        elif "context_seg_masks" in data and "context_seg_infos" in data:
            self.save_seg_masks(
                mode="context",
                seg_masks=data["context_seg_masks"],
                seg_info=data["context_seg_infos"],
                save_dir=save_dir,
                scene_names=scene_names,
                context_views_id=context_views_id,
            )
            self.save_gt_seg_masks(
                mode="context",
                mask_labels=data["context_mask_labels"],
                class_labels=data["context_class_labels"],
                save_dir=save_dir,
                scene_names=scene_names,
                context_views_id=context_views_id,
            )
            self.draw_overlay_segm_masks(
                gt_images=context_views_images,
                seg_masks=data["context_seg_masks"],
                seg_infos=data["context_seg_infos"],
                mask_labels=data["context_mask_labels"],
                class_labels=data["context_class_labels"],
                save_dir=save_dir,
                scene_names=scene_names,
                context_views_id=context_views_id,
            )

    def write_files(self):
        num_stashed = len(self.pool)
        if num_stashed == 1:
            self.write_file(self.pool[0])
        elif num_stashed > 1:
            log.info(f"Writing {num_stashed} data to disk...")
            with ThreadPoolExecutor(max_workers=os.cpu_count() // 8) as executor:
                list(
                    executor.map(
                        self.write_file,
                        self.pool,
                    )
                )
        self.reset()

    def visualize_recon_image(
        self,
        rendered_images: Tensor,  # (B, N, C, H, W)
        rendered_depths: Tensor,  # (B, N, H, W)
        gt_images: Tensor,  # (B, N, C, H, W)
        gt_depths: Tensor,  # (B, N, H, W)
        save_dir: str,
        scene_names: list[str],  # [B]
        context_views_id: list[int],  # [B]
        target_views_id: list[int],  # [B*[N]]
    ):
        B, N, C, H, W = rendered_images.shape
        rendered_images = [
            img.permute(0, 2, 3, 1).detach().cpu().numpy() for img in rendered_images
        ]
        rendered_images = [img * 255 for img in rendered_images]
        rendered_depths = [img.detach().cpu().unsqueeze(1) for img in rendered_depths]
        if self.cfg.log_colored_depth:
            color_map = kornia.color.ColorMap(kornia.color.ColorMapType.jet)
            rendered_depths_color = [
                1
                - (img.log() - img[img > 0][:16_000_000].quantile(0.01).log())
                / (
                    img.view(-1)[:16_000_000].quantile(0.99).log()
                    - img[img > 0][:16_000_000].quantile(0.01).log()
                )
                for img in rendered_depths
            ]
            rendered_depths_color = [
                kornia.color.apply_colormap(img, color_map)
                for img in rendered_depths_color
            ]
            rendered_depths_color = [
                img.permute(0, 2, 3, 1).numpy() for img in rendered_depths_color
            ]
            rendered_depths_color = [img * 255 for img in rendered_depths_color]
        rendered_depths = [img.squeeze(1).numpy() * 1000 for img in rendered_depths]
        rendered_depths = [img.astype(np.int32) for img in rendered_depths]

        gt_images = [
            img.permute(0, 2, 3, 1).detach().cpu().numpy() for img in gt_images
        ]
        gt_images = [img * 255 for img in gt_images]
        gt_depths = [img.detach().cpu().unsqueeze(1) for img in gt_depths]
        if self.cfg.log_colored_depth:
            gt_depths_color = [
                (img - img.min()) / (img.max() - img.min()) for img in gt_depths
            ]
            gt_depths_color = [
                kornia.color.apply_colormap(img, color_map) for img in gt_depths_color
            ]
            gt_depths_color = [
                img.permute(0, 2, 3, 1).numpy() for img in gt_depths_color
            ]
            gt_depths_color = [img * 255 for img in gt_depths_color]
        gt_depths = [img.squeeze(1).numpy() * 1000 for img in gt_depths]
        gt_depths = [img.astype(np.int32) for img in gt_depths]

        for i in range(B):
            scene_name = scene_names[i]
            context_view_id = context_views_id[i]
            context_view_str = "_".join(map(str, context_view_id))
            save_subdir = f"{save_dir}/{scene_name}_context{context_view_str}"
            os.makedirs(save_subdir, exist_ok=True)
            if os.path.exists(f"{save_subdir}/rgb"):
                continue
            os.makedirs(f"{save_subdir}/rgb", exist_ok=True)
            os.makedirs(f"{save_subdir}/rgb_gt", exist_ok=True)
            os.makedirs(f"{save_subdir}/depth", exist_ok=True)
            os.makedirs(f"{save_subdir}/depth_gt", exist_ok=True)
            if self.cfg.log_colored_depth:
                os.makedirs(f"{save_subdir}/depth_color", exist_ok=True)
                os.makedirs(f"{save_subdir}/depth_gt_color", exist_ok=True)
            for j in range(N):
                rendered_image = rendered_images[i][j]
                rendered_depth = rendered_depths[i][j]
                gt_image = gt_images[i][j]
                gt_depth = gt_depths[i][j]
                target_view_id = target_views_id[i][j]
                rendered_image_path = Path(
                    f"{save_subdir}/rgb/{scene_name}_{target_view_id}.png"
                )
                rendered_image = Image.fromarray(rendered_image.astype(np.uint8))
                rendered_image.save(rendered_image_path)

                rendered_depth_path = Path(
                    f"{save_subdir}/depth/{scene_name}_{target_view_id}.png"
                )
                rendered_depth = Image.fromarray(rendered_depth)
                rendered_depth.save(rendered_depth_path)

                gt_image_path = Path(
                    f"{save_subdir}/rgb_gt/{scene_name}_{target_view_id}.png"
                )
                gt_image = Image.fromarray(gt_image.astype(np.uint8))
                gt_image.save(gt_image_path)

                gt_depth_path = Path(
                    f"{save_subdir}/depth_gt/{scene_name}_{target_view_id}.png"
                )
                gt_depth = Image.fromarray(gt_depth)
                gt_depth.save(gt_depth_path)
                if self.cfg.log_colored_depth:
                    rendered_depth_color = rendered_depths_color[i][j]
                    gt_depth_color = gt_depths_color[i][j]
                    path = Path(
                        f"{save_subdir}/depth_color/{scene_name}_{target_view_id}.png"
                    )
                    rendered_depth_color = Image.fromarray(
                        rendered_depth_color.astype(np.uint8)
                    )
                    rendered_depth_color.save(path)
                    path = Path(
                        f"{save_subdir}/depth_gt_color/{scene_name}_{target_view_id}.png"
                    )
                    gt_depth_color = Image.fromarray(gt_depth_color.astype(np.uint8))
                    gt_depth_color.save(path)

    def save_gaussians(
        self,
        gaussians: Gaussians,
        save_dir: str,
        scene_names: list[str],  # [B*2]
        context_views: list[str],  # [B*2]
    ):
        B, N = gaussians.means.shape[:2]
        for i in range(B):
            means = gaussians.means[i, ...]
            scales = gaussians.scales[i, ...]
            rotations = gaussians.rotations[i, ...]
            opacities = gaussians.opacities[i, ...]
            harmonics = gaussians.harmonics[i, ...]
            semantic_labels = (
                gaussians.semantic_labels[i, ...]
                if hasattr(gaussians, "semantic_labels")
                else None
            )
            instance_labels = (
                gaussians.instance_labels[i, ...]
                if hasattr(gaussians, "instance_labels")
                else None
            )
            seg_query_class_logits = (
                gaussians.seg_query_class_logits[i]
                if hasattr(gaussians, "seg_query_class_logits")
                else None
            )
            context_views_str = "_".join(map(str, context_views[i]))
            save_subdir = f"{save_dir}/{scene_names[i]}_context{context_views_str}"
            os.makedirs(save_subdir, exist_ok=True)
            if os.path.exists(
                f"{save_subdir}/{scene_names[i]}_context{context_views_str}.ply"
            ):
                continue
            export_ply(
                means=means,
                scales=scales,
                rotations=rotations,
                harmonics=harmonics,
                opacities=opacities,
                semantic_labels=semantic_labels,
                instance_labels=instance_labels,
                seg_query_class_logits=seg_query_class_logits,
                path=Path(
                    f"{save_subdir}/{scene_names[i]}_context{context_views_str}.ply"
                ),
                save_sh_dc_only=self.cfg.save_sh_dc_only,
            )

    def save_seg_masks(
        self,
        mode: Literal["context", "target"],
        seg_masks: list[Tensor],
        seg_info: list[list[dict]],
        save_dir: str,
        scene_names: list[str],  # [B]
        context_views_id: list[list[int]],  # [B]
        target_views_id: list[list[int]] = None,  # [B*[N]]
    ):
        for idx, (
            seg_mask,
            seg,
            scene_name,
            context_view_id,
        ) in enumerate(
            zip(
                seg_masks,
                seg_info,
                scene_names,
                context_views_id,
            )
        ):
            context_view_str = "_".join(map(str, context_views_id[idx]))
            base_path = f"{save_dir}/{scene_name}_context{context_view_str}"
            os.makedirs(base_path, exist_ok=True)
            if os.path.exists(f"{base_path}/{mode}_seg_pred/pred.json"):
                continue
            os.makedirs(f"{base_path}/{mode}_seg_pred", exist_ok=True)
            N, H, W = seg_mask.shape
            instance_id = seg_mask.clone()
            semantic_id = torch.zeros_like(instance_id)
            for info in seg:
                semantic_id[seg_mask == info["id"]] = info["label_id"] + 1
            seg_save = deepcopy(seg)
            for info in seg_save:
                info["label_id"] += 1
            json.dump(
                seg_save, open(f"{base_path}/{mode}_seg_pred/pred.json", "w"), indent=4
            )
            instance_id[semantic_id == 0] = 0
            segment_id = 1000 * semantic_id + instance_id
            if mode == "context":
                views_id = context_view_id
            elif mode == "target":
                views_id = target_views_id[idx]
            else:
                raise ValueError(f"Unknown mode: {mode}")
            for seg_id, vid in zip(segment_id, views_id):
                segment_map = torch.zeros(3, H, W, dtype=torch.uint8)
                segment_map[0, ...] = seg_id % 256
                segment_map[1, ...] = seg_id // 256
                segment_map[2, ...] = seg_id // 256 // 256
                segment_map = (
                    segment_map.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                )
                path = Path(f"{base_path}/{mode}_seg_pred/{scene_name}_pred{vid}.png")
                segment_map = Image.fromarray(segment_map)
                segment_map.save(path)

    def save_gt_seg_masks(
        self,
        mode: Literal["context", "target"],
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
        save_dir: str,
        scene_names: list[str],  # [B]
        context_views_id: list[list[int]],  # [B]
        target_views_id: list[list[int]] = None,  # [B*[N]]
    ):
        for idx, (
            mask_label,
            class_label,
            scene_name,
            context_view_id,
        ) in enumerate(
            zip(
                mask_labels,
                class_labels,
                scene_names,
                context_views_id,
            )
        ):
            _, N, H, W = mask_label.shape
            context_view_str = "_".join(map(str, context_views_id[idx]))
            base_path = f"{save_dir}/{scene_name}_context{context_view_str}"
            os.makedirs(f"{base_path}/{mode}_seg_gt", exist_ok=True)
            gt_instance_id = torch.zeros(N, H, W, dtype=torch.int64)
            gt_semantic_id = torch.zeros(N, H, W, dtype=torch.int64)
            for ins_id, (mask, cls) in enumerate(zip(mask_label, class_label)):
                gt_instance_id[mask == 1] = ins_id + 1
                gt_semantic_id[mask == 1] = cls + 1
            gt_segment_id = 1000 * gt_semantic_id + gt_instance_id
            if mode == "context":
                views_id = context_view_id
            elif mode == "target":
                views_id = target_views_id[idx]
            else:
                raise ValueError(f"Unknown mode: {mode}")
            for gt_seg, vid in zip(gt_segment_id, views_id):
                gt_segment_map = torch.zeros(3, H, W, dtype=torch.uint8)
                gt_segment_map[0, ...] = gt_seg % 256
                gt_segment_map[1, ...] = gt_seg // 256
                gt_segment_map[2, ...] = gt_seg // 256 // 256
                gt_segment_map = (
                    gt_segment_map.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                )
                path = Path(f"{base_path}/{mode}_seg_gt/{scene_name}_gt{vid}.png")
                gt_segment_map = Image.fromarray(gt_segment_map)
                gt_segment_map.save(path)

    def draw_overlay_segm_masks(
        self,
        gt_images: Tensor,  # (B, N, C, H, W)
        seg_masks: list[Tensor],  # (B, N, H, W)
        seg_infos: list[list[dict]],  # (B, N, H, W)
        mask_labels: list[Tensor],  # (B, N, H, W)
        class_labels: list[Tensor],  # (B, N, H, W)
        save_dir: str,
        scene_names: list[str],  # [B]
        context_views_id: list[list[int]],  # [B]
        target_views_id: list[list[int]] = None,  # [B*[N]]
        text: list[str] = None,  # [B*[N]]
    ):
        try:
            B, N, C, H, W = gt_images.shape
            images = [
                (img.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                for img in gt_images
            ]  # [(2, H, W, C)] * B
            masks = [m.cpu().numpy() for m in seg_masks]  # [(2, H, W)] * B
            overlayed_images = []
            for i, (image, mask, segment_info) in enumerate(
                zip(images, masks, seg_infos)
            ):
                # image: (2, H, W, C), mask: (2, H, W), segment_info: [{"id": int, "label_id": int, "score": float}]*num_instances
                colored_masks = []
                for im, mas in zip(image, mask):
                    # im: (H, W, C), mas: (H, W)
                    colored_mask = np.zeros_like(im, dtype=np.uint8)  # (H, W, C)
                    color = None
                    for seg_info in segment_info:
                        semantic_id = self.continuous2semantic[seg_info["label_id"] + 1]
                        if semantic_id == 0:
                            continue
                        color = self.color_palette[semantic_id]
                        # draw mask
                        colored_mask[mas == seg_info["id"]] = color  # (H, W, C)
                    for seg_info in segment_info:
                        semantic_id = self.continuous2semantic[seg_info["label_id"] + 1]
                        if semantic_id == 0:
                            continue
                        color = self.color_palette[semantic_id]
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
                        category_text = f"{seg_info['id']}|{self.semantic2name[semantic_id]}|{seg_info['score']:.2f}"
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
                    self.cfg.overlay_mask_alpha * colored_masks[colored_masks != 0]
                    + (1 - self.cfg.overlay_mask_alpha)
                    * overlayed_image[colored_masks != 0]
                )
                overlayed_images.append(overlayed_image)
            batch_gt_images = []

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
                        semantic_id = self.continuous2semantic[cls + 1]
                        if semantic_id == 0:
                            continue
                        color = self.color_palette[semantic_id]
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
                        category_text = f"{self.semantic2name[semantic_id]}"
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
                context_views_str = "_".join(map(str, context_views_id[i]))
                if target_views_id is not None:
                    target_views_str = "_".join(map(str, target_views_id[i]))
                else:
                    target_views_str = context_views_str
                os.makedirs(
                    f"{save_dir}/{scene_name}_context{context_views_str}", exist_ok=True
                )
                path = f"{save_dir}/{scene_name}_context{context_views_str}/{scene_name}_seg{target_views_str}.png"
                if os.path.exists(path):
                    continue
                if text is not None:
                    overlayed_text = text[i]
                    with open(path.replace(".png", ".txt"), "w") as f:
                        f.write(str(overlayed_text))
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
            # raise e

    def save_seg_ids(
        self,
        mode: Literal["context", "target"],
        semantic_ids: Tensor,  # (B, N, H, W)
        instance_ids: Tensor,  # (B, N, H, W)
        save_dir: str,
        scene_names: list[str],  # [B]
        context_views_id: list[list[int]],  # [B]
        target_views_id: list[list[int]] = None,  # [B*[N]]
        seg_infos: list[list[dict]] = None,
    ):
        B, N, H, W = semantic_ids.shape
        for idx, (
            sem_id,
            ins_id,
            scene_name,
            context_view_id,
        ) in enumerate(
            zip(
                semantic_ids,
                instance_ids,
                scene_names,
                context_views_id,
            )
        ):
            context_view_str = "_".join(map(str, context_view_id))
            base_path = f"{save_dir}/{scene_name}_context{context_view_str}"
            os.makedirs(base_path, exist_ok=True)
            os.makedirs(f"{base_path}/{mode}_seg_pred", exist_ok=True)
            seg_save = deepcopy(seg_infos[idx])
            json.dump(
                seg_save, open(f"{base_path}/{mode}_seg_pred/pred.json", "w"), indent=4
            )
            segment_id = 1000 * sem_id + ins_id
            if mode == "context":
                views_id = context_view_id
            elif mode == "target":
                views_id = target_views_id[idx]
            else:
                raise ValueError(f"Unknown mode: {mode}")
            for seg_id, vid in zip(segment_id, views_id):
                segment_map = torch.zeros(3, H, W, dtype=torch.uint8)
                segment_map[0, ...] = seg_id % 256
                segment_map[1, ...] = seg_id // 256
                segment_map[2, ...] = seg_id // 256 // 256
                segment_map = (
                    segment_map.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                )
                path = Path(f"{base_path}/{mode}_seg_pred/{scene_name}_pred{vid}.png")
                segment_map = Image.fromarray(segment_map)
                segment_map.save(path)

    def draw_overlay_segm_ids(
        self,
        gt_images: Tensor,  # (B, N, C, H, W)
        semantic_ids: Tensor,  # (B, N, H, W)
        instance_ids: Tensor,  # (B, N, H, W)
        mask_labels: list[Tensor],  # (B, N, H, W)
        class_labels: list[Tensor],  # (B, N, H, W)
        save_dir: str,
        scene_names: list[str],  # [B]
        context_views_id: list[list[int]],  # [B]
        target_views_id: list[list[int]] = None,  # [B*[N]]
    ):
        try:
            B, N, C, H, W = gt_images.shape
            images = [
                (img.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                for img in gt_images
            ]  # [(N, H, W, C)] * B
            masks = torch.stack([semantic_ids, instance_ids], dim=-1)  # (B, N, H, W, 2)
            masks = masks.permute(0, 1, 4, 2, 3)  # (B, N, 2, H, W)
            masks = [m.cpu().numpy() for m in masks]  # [(N, 2, H, W)] * B
            overlayed_images = []
            for i, (image, mask) in enumerate(zip(images, masks)):
                # image: (N, H, W, C), mask: (N, 2, H, W)
                colored_masks = []
                for im, mas in zip(image, mask):
                    # im: (H, W, C), mas: (2, H, W)
                    colored_mask = np.zeros_like(im, dtype=np.uint8)  # (H, W, C)
                    color = None
                    reshaped_mas = mas.reshape(2, -1).T  # (H*W, 2)
                    unique_values, indices = np.unique(
                        reshaped_mas, axis=0, return_inverse=True
                    )
                    for unique_value in unique_values:
                        seg_id, ins_id = unique_value
                        if seg_id == 0:
                            continue
                        semantic_id = self.continuous2semantic[seg_id]
                        color = self.color_palette[semantic_id]
                        # draw mask
                        colored_mask[mas[0] == seg_id] = color  # (H, W, C)
                    for unique_value in unique_values:
                        seg_id, ins_id = unique_value
                        if seg_id == 0:
                            continue
                        semantic_id = self.continuous2semantic[seg_id]
                        color = self.color_palette[semantic_id]
                        mask_area = mas[1] == ins_id  # (H, W)
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
                        category_text = f"{ins_id}|{self.semantic2name[semantic_id]}"
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
                overlayed_image = image.transpose(1, 0, 2, 3).reshape(H, N * W, C)
                overlayed_image[colored_masks != 0] = (
                    self.cfg.overlay_mask_alpha * colored_masks[colored_masks != 0]
                    + (1 - self.cfg.overlay_mask_alpha)
                    * overlayed_image[colored_masks != 0]
                )
                overlayed_images.append(overlayed_image)
            batch_gt_images = []
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
                        semantic_id = self.continuous2semantic[cls + 1]
                        if semantic_id == 0:
                            continue
                        color = self.color_palette[semantic_id]
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
                        category_text = f"{self.semantic2name[semantic_id]}"
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
                context_views_str = "_".join(map(str, context_views_id[i]))
                if target_views_id is not None:
                    target_views_str = "_".join(map(str, target_views_id[i]))
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
