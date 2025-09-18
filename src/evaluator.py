import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import hydra
from omegaconf import DictConfig
from src.config import RootCfg, load_typed_root_config
import torch
import json
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from src.utils.miou import MeanIoU
from torchmetrics.detection import MeanAveragePrecision, PanopticQuality
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from src.config import EvaluatorCfg
from src.utils.tensor_utils import itemize
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class Evaluator:
    def __init__(self, cfg: EvaluatorCfg):
        self.cfg = cfg
        self.device = cfg.device
        self.things = [
            thing + 1 for thing in self.cfg.things
        ]  # align with 1-based index
        self.stuffs = [
            stuff + 1 for stuff in self.cfg.stuffs
        ]  # align with 1-based index
        self.eval_context_seg = (
            self.cfg.eval_context_miou
            or self.cfg.eval_context_pq
            or self.cfg.eval_context_map
        )
        self.eval_target_seg = (
            self.cfg.eval_target_miou
            or self.cfg.eval_target_pq
            or self.cfg.eval_target_map
        )

    def setup(self):
        if self.cfg.eval_image_quality:
            self.psnr = PeakSignalNoiseRatio(sync_on_compute=False).to(self.device)
            self.ssim = StructuralSimilarityIndexMeasure(sync_on_compute=False).to(
                self.device
            )
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                "vgg", normalize=True, sync_on_compute=False
            ).to(self.device)
            self.target_psnr = []
            self.target_ssim = []
            self.target_lpips = []
        if self.cfg.eval_context_miou:
            self.context_miou = MeanIoU(
                num_classes=len(self.cfg.id2label) + 1,
                include_background=False,
                input_format="index",
                per_class=True,
                sync_on_compute=False,
            ).to(self.device)
        if self.cfg.eval_target_miou:
            self.target_miou = MeanIoU(
                num_classes=len(self.cfg.id2label) + 1,
                include_background=False,
                input_format="index",
                per_class=True,
                sync_on_compute=False,
            ).to(self.device)
        if self.cfg.eval_context_pq:
            self.context_pq = PanopticQuality(
                things=self.things,
                stuffs=self.stuffs,
                return_per_class=True,
                allow_unknown_preds_category=True,
                sync_on_compute=False,
            ).to(self.device)
        if self.cfg.eval_target_pq:
            self.target_pq = PanopticQuality(
                things=self.things,
                stuffs=self.stuffs,
                return_per_class=True,
                allow_unknown_preds_category=True,
                sync_on_compute=False,
            ).to(self.device)
        if self.cfg.eval_context_map:
            self.context_mAP = MeanAveragePrecision(
                iou_type="segm",
                class_metrics=True,
                sync_on_compute=False,
            ).to(self.device)
            self.context_map_preds, self.context_map_gts = [], []
        if self.cfg.eval_target_map:
            self.target_mAP = MeanAveragePrecision(
                iou_type="segm",
                class_metrics=True,
                sync_on_compute=False,
            ).to(self.device)
            self.target_map_preds, self.target_map_gts = [], []
        if self.cfg.eval_depth_quality:
            self.target_absrels = []
            self.target_rmses = []

    def load_image_to_tensor(self, image_path, normalize=True):
        img = np.array(Image.open(image_path)).astype(np.float32)
        if normalize:
            img /= 255.0
        if img.ndim == 2:
            return torch.from_numpy(img).to(self.device)
        elif img.ndim == 3:
            return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def process_segmentation(self, seg_pred_dir, seg_gt_dir):
        seg_pred_items = sorted([item for item in seg_pred_dir.glob("*.png")])
        pred_semantics, pred_instances, gt_semantics, gt_instances = [], [], [], []

        for item in seg_pred_items:
            # Load predicted segmentation
            pred_img = torch.from_numpy(np.array(Image.open(item))).to(self.device)
            pred_segment_id = (
                pred_img[:, :, 0].to(torch.int64)
                + pred_img[:, :, 1].to(torch.int64) * 256
                + pred_img[:, :, 2].to(torch.int64) * 256 * 256
            )
            pred_semantics.append(pred_segment_id // 1000)
            pred_instances.append(pred_segment_id % 1000)

            # Load ground truth segmentation
            gt_item = seg_gt_dir / item.name.replace("pred", "gt")
            gt_img = torch.from_numpy(np.array(Image.open(gt_item))).to(self.device)
            gt_segment_id = (
                gt_img[:, :, 0].to(torch.int64)
                + gt_img[:, :, 1].to(torch.int64) * 256
                + gt_img[:, :, 2].to(torch.int64) * 256 * 256
            )
            gt_semantics.append(gt_segment_id // 1000)
            gt_instances.append(gt_segment_id % 1000)

        # Concatenate along height dimension
        pred_semantics = torch.cat(pred_semantics, dim=0).unsqueeze(0)  # (1, N*H, W)
        pred_instances = torch.cat(pred_instances, dim=0).unsqueeze(0)
        gt_semantics = torch.cat(gt_semantics, dim=0).unsqueeze(0)
        gt_instances = torch.cat(gt_instances, dim=0).unsqueeze(0)

        # Prepare ground truth for MeanAveragePrecision
        gt_unique = torch.unique(gt_instances)
        gt_masks, gt_labels = [], []
        for gt_id in gt_unique:
            if gt_id == 0:  # Skip background
                continue
            gt_mask = gt_instances == gt_id
            gt_label = gt_semantics[gt_mask][0] - 1  # Adjust labels (0-based)
            if gt_label + 1 in self.stuffs:
                continue
            gt_masks.append(gt_mask)
            gt_labels.append(gt_label)
        gt_masks = (
            torch.concat(gt_masks, dim=0).to(self.device)
            if gt_masks
            else torch.zeros((0, *gt_instances.shape[1:]), dtype=torch.bool).to(
                self.device
            )
        )
        gt_labels = (
            torch.tensor(gt_labels).to(self.device)
            if gt_labels
            else torch.zeros(0, dtype=torch.long).to(self.device)
        )
        exist_json = False
        if os.path.exists(seg_pred_dir / "pred.json"):
            # Load prediction JSON
            with open(seg_pred_dir / "pred.json", "r") as f:
                preds = json.load(f)
            exist_json = True

        # Prepare predictions for MeanAveragePrecision
        pred_unique = torch.unique(pred_instances)
        pred_masks, pred_labels, pred_scores = [], [], []
        for pred_id in pred_unique:
            if pred_id == 0:
                continue
            instance_mask = pred_instances == pred_id
            pred_masks.append(instance_mask)
            if not exist_json:
                pred_labels.append(pred_semantics[instance_mask][0] - 1)
                pred_scores.append(1.0)
            else:
                pred_info = [info for info in preds if info["id"] == pred_id.item()]
                if pred_info:
                    pred_labels.append(pred_info[0]["label_id"] - 1)  # Adjust labels
                    pred_scores.append(np.mean([info["score"] for info in pred_info]))
        pred_masks = (
            torch.concat(pred_masks, dim=0).to(self.device)
            if pred_masks
            else torch.zeros((0, *gt_instances.shape[1:]), dtype=torch.bool).to(
                self.device
            )
        )
        pred_labels = (
            torch.tensor(pred_labels).to(self.device)
            if pred_labels
            else torch.zeros(0, dtype=torch.long).to(self.device)
        )
        pred_scores = (
            torch.tensor(pred_scores).to(self.device)
            if pred_scores
            else torch.zeros(0, dtype=torch.float).to(self.device)
        )
        return {
            "pred_semantics": pred_semantics,
            "pred_instances": pred_instances,
            "gt_semantics": gt_semantics,
            "gt_instances": gt_instances,
            "map_pred": {
                "scores": pred_scores,
                "labels": pred_labels,
                "masks": pred_masks,
            },
            "map_gt": {"labels": gt_labels, "masks": gt_masks},
        }

    def fit_scale_and_shift(self, pred, gt):
        valid_mask = gt > 0
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]
        A = torch.stack([pred_valid, torch.ones_like(pred_valid)], dim=1)
        b = gt_valid
        scale, shift = torch.linalg.lstsq(A, b).solution
        return scale, shift

    def evaluate(self, path, eval_scan_num=-1):
        eval_path = Path(path)
        log.info(f"Evaluating results from: {eval_path} ...")
        scene_dirs = sorted([d for d in eval_path.iterdir() if d.is_dir()])
        scene_dirs = scene_dirs[:eval_scan_num] if eval_scan_num > 0 else scene_dirs
        log.info(f"Found {len(scene_dirs)} scenes")
        for scene_dir in tqdm(scene_dirs, desc="Evaluating scenes"):
            scene_info = {
                "scan": scene_dir.name.split("_context")[0],
                "context_ids": list(
                    map(int, scene_dir.name.split("_context")[1].split("_"))
                ),
            }
            if self.cfg.eval_image_quality:
                rgb_dir = scene_dir / "rgb"
                rgb_gt_dir = scene_dir / "rgb_gt"
                items = sorted(rgb_dir.glob("*.png"))
                render_scores = []
                for item in items:
                    rgb = self.load_image_to_tensor(item)
                    rgb_gt = self.load_image_to_tensor(rgb_gt_dir / item.name)
                    result = {
                        "item": item.name,
                        "psnr": self.psnr(rgb, rgb_gt).item(),
                        "ssim": self.ssim(rgb, rgb_gt).item(),
                        "lpips": self.lpips(rgb, rgb_gt).item(),
                    }
                    render_scores.append(result)
                    self.target_psnr.append(result["psnr"])
                    self.target_ssim.append(result["ssim"])
                    self.target_lpips.append(result["lpips"])
                with open(scene_dir / "render_scores.json", "w") as f:
                    json.dump(render_scores, f, indent=4)
            if self.eval_context_seg:
                context_seg_pred_dir = scene_dir / "context_seg_pred"
                context_seg_gt_dir = scene_dir / "context_seg_gt"
                context_data = self.process_segmentation(
                    context_seg_pred_dir, context_seg_gt_dir
                )
                if self.cfg.eval_context_miou:
                    self.context_miou.update(
                        context_data["pred_semantics"],
                        context_data["gt_semantics"],
                    )
                if self.cfg.eval_context_pq:
                    self.context_pq.update(
                        torch.stack(
                            [
                                context_data["pred_semantics"],
                                context_data["pred_instances"],
                            ],
                            dim=-1,
                        ),
                        torch.stack(
                            [
                                context_data["gt_semantics"],
                                context_data["gt_instances"],
                            ],
                            dim=-1,
                        ),
                    )
                if self.cfg.eval_context_map:
                    self.context_map_preds.append(context_data["map_pred"])
                    self.context_map_gts.append(context_data["map_gt"])
            if self.eval_target_seg:
                target_seg_pred_dir = scene_dir / "target_seg_pred"
                target_seg_gt_dir = scene_dir / "target_seg_gt"
                target_data = self.process_segmentation(
                    target_seg_pred_dir, target_seg_gt_dir
                )
                if self.cfg.eval_target_miou:
                    self.target_miou.update(
                        target_data["pred_semantics"],
                        target_data["gt_semantics"],
                    )
                if self.cfg.eval_target_pq:
                    self.target_pq.update(
                        torch.stack(
                            [
                                target_data["pred_semantics"],
                                target_data["pred_instances"],
                            ],
                            dim=-1,
                        ),
                        torch.stack(
                            [
                                target_data["gt_semantics"],
                                target_data["gt_instances"],
                            ],
                            dim=-1,
                        ),
                    )
                if self.cfg.eval_target_map:
                    self.target_map_preds.append(target_data["map_pred"])
                    self.target_map_gts.append(target_data["map_gt"])
            if self.cfg.eval_depth_quality:
                depth_dir = scene_dir / "depth"
                depth_gt_dir = scene_dir / "depth_gt"
                items = sorted(depth_dir.glob("*.png"))
                depth_scores = []
                for item in items:
                    depth = self.load_image_to_tensor(item, normalize=False) / 1000.0
                    depth_gt = (
                        self.load_image_to_tensor(
                            depth_gt_dir / item.name, normalize=False
                        )
                        / 1000.0
                    )
                    scale, shift = self.fit_scale_and_shift(depth, depth_gt)
                    depth_scaled = depth * scale + shift
                    absrel = torch.mean(
                        torch.abs(depth_scaled[depth_gt > 0] - depth_gt[depth_gt > 0])
                        / depth_gt[depth_gt > 0]
                    )
                    rmse = torch.sqrt(
                        torch.mean(
                            (depth_scaled[depth_gt > 0] - depth_gt[depth_gt > 0]) ** 2
                        )
                    )
                    result = {
                        "item": item.name,
                        "absrel": absrel.item(),
                        "rmse": rmse.item(),
                    }
                    depth_scores.append(result)
                    self.target_absrels.append(absrel.item())
                    self.target_rmses.append(rmse.item())
                with open(scene_dir / "depth_scores.json", "w") as f:
                    json.dump(depth_scores, f, indent=4)
        log.info("Evaluating across all scenes ...")
        result = {}
        if self.cfg.eval_image_quality:
            result["psnr"] = itemize(np.mean(self.target_psnr))
            result["ssim"] = itemize(np.mean(self.target_ssim))
            result["lpips"] = itemize(np.mean(self.target_lpips))
        if self.cfg.eval_depth_quality:
            result["absrel"] = itemize(np.mean(self.target_absrels))
            result["rmse"] = itemize(np.mean(self.target_rmses))
        if self.cfg.eval_context_miou:
            result["context_ious_per_class"] = itemize(self.context_miou.compute())
            result["context_miou"] = itemize(np.mean(result["context_ious_per_class"]))
        if self.cfg.eval_target_miou:
            result["target_ious_per_class"] = itemize(self.target_miou.compute())
            result["target_miou"] = itemize(np.mean(result["target_ious_per_class"]))
        if self.cfg.eval_context_pq:
            result["context_pqs_per_class"] = itemize(self.context_pq.compute())
            result["context_pq"] = itemize(np.mean(result["context_pqs_per_class"]))
        if self.cfg.eval_target_pq:
            result["target_pqs_per_class"] = itemize(self.target_pq.compute())
            result["target_pq"] = itemize(np.mean(result["target_pqs_per_class"]))
        if self.cfg.eval_context_map:
            self.context_mAP.update(
                self.context_map_preds,
                self.context_map_gts,
            )
            result["context_map"] = itemize(self.context_mAP.compute())
        if self.cfg.eval_target_map:
            self.target_mAP.update(
                self.target_map_preds,
                self.target_map_gts,
            )
            result["target_map"] = itemize(self.target_mAP.compute())
        log.info(f"Evaluation result: {result}")
        log.info(f"Saving evaluation result to {eval_path / 'results.json'} ...")
        with open(eval_path / "results.json", "w") as f:
            json.dump(result, f, indent=4)
        return result


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="main",
)
def eval_main(cfg: DictConfig):
    cfg: RootCfg = load_typed_root_config(cfg)
    evaluator = Evaluator(cfg.pipeline.evaluator)
    evaluator.setup()
    evaluator.evaluate(cfg.pipeline.evaluator.eval_path)


if __name__ == "__main__":
    from lightning.pytorch.utilities import rank_zero_only

    rank_zero_only.rank = 0
    eval_main()
