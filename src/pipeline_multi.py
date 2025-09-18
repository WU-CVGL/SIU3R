import torch
from lightning import LightningModule
from src.config import RootCfg
from src.models.model import SIU3RModel
from src.models.model_multi import SIU3RMultiViewModel
from src.models.gaussian_renderer import SplattingCUDA
from src.visualizer import Visualizer
from src.evaluator import Evaluator
from torchmetrics import MeanSquaredError
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
import os
from torch import Tensor
import torch.nn.functional as F

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PipelineMultiView(LightningModule):
    def __init__(self, cfg: RootCfg):
        super().__init__()
        self.cfg = cfg
        self.pipecfg = cfg.pipeline
        self.model = SIU3RMultiViewModel(self.pipecfg.model)
        self.gaussian_renderer = SplattingCUDA()
        self.mse_loss = MeanSquaredError()
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.lpips = LearnedPerceptualImagePatchSimilarity("vgg", normalize=True)
        self.lpips.eval()
        self.lpips.requires_grad_(False)
        self.visualizer = Visualizer(self.pipecfg.visualizer)
        self.save_hyperparameters()

    def setup(self, stage: str):
        if self.cfg.mode == "train":
            self.model.load_recon_ckpt()
            self.model.load_seg_ckpt()
        self.vidx, self.hidx, self.widx = torch.meshgrid(
            torch.arange(self.cfg.datamodule.dataset_cfg.num_extra_target_views + 2),
            torch.arange(self.pipecfg.model.image_size[0]),
            torch.arange(self.pipecfg.model.image_size[1]),
            indexing="ij",
        )
        self.vidx = self.vidx.to(self.device)
        self.hidx = self.hidx.to(self.device)
        self.widx = self.widx.to(self.device)

    def step_wo_lift(self, batch, batch_idx):
        context_mask_labels = batch["context_mask_labels"]
        context_class_labels = batch["context_class_labels"]
        context_views_images = batch["context_views_images"]
        context_views_intrinsics = batch["context_views_intrinsics"][:, :, :3, :3]
        target_views_intrinsics = batch["target_views_intrinsics"][:, :, :3, :3]
        target_views_extrinsics = batch["target_views_extrinsics"]

        (
            gaussians,
            context_seg_output,
            context_seg_masks,
            context_seg_infos,
        ) = self.model(
            context_views_images,
            context_views_intrinsics,
            context_mask_labels,
            context_class_labels,
        )
        render_output = self.gaussian_renderer(
            gaussians=gaussians,
            extrinsics=target_views_extrinsics,
            intrinsics=target_views_intrinsics,
            image_shape=self.pipecfg.model.image_size,
            render_color=True,
        )
        return (
            gaussians,
            context_seg_output,
            context_seg_masks,
            context_seg_infos,
            render_output,
        )

    def step_w_query_class_logit_lift(self, batch, batch_idx):
        context_mask_labels = batch["context_mask_labels"]
        context_class_labels = batch["context_class_labels"]
        context_views_images = batch["context_views_images"]
        context_views_intrinsics = batch["context_views_intrinsics"][:, :, :3, :3]
        target_views_intrinsics = batch["target_views_intrinsics"][:, :, :3, :3]
        target_views_extrinsics = batch["target_views_extrinsics"]
        target_mask_labels = batch["target_mask_labels"]
        target_class_labels = batch["target_class_labels"]
        B, V, _, H, W = context_views_images.shape

        context_ids = batch["context_views_id"]
        target_ids = batch["target_views_id"]
        context_in_target_ids = []
        for context_id, target_id in zip(context_ids, target_ids):
            context_in_target_id = []
            for pos, idx in enumerate(target_id):
                if idx in context_id:
                    context_in_target_id.append(pos)
            context_in_target_ids.append(context_in_target_id)

        (
            gaussians,
            context_seg_output,
            context_seg_masks,
            context_seg_infos,
            context_seg_query_scores,
        ) = self.model.forward(
            context_views_images,
            context_views_intrinsics,
            context_mask_labels,
            context_class_labels,
            enable_query_class_logit_lift=True,
        )
        for idx, segq in enumerate(gaussians.seg_query_class_logits):
            if segq.shape[0] != V * H * W:
                print(
                    f"segq shape {segq.shape} is not {V * H * W}, this may cause issues in rendering."
                )
                print(context_ids[idx])
        render_output = self.gaussian_renderer(
            gaussians=gaussians,
            extrinsics=target_views_extrinsics,
            intrinsics=target_views_intrinsics,
            image_shape=self.pipecfg.model.image_size,
            render_color=True,
            render_qc_logits=True,
        )

        render_qc_logits = render_output["render_qc_logits"]
        all_sem_id = []
        all_ins_id = []
        seg_infos = []
        # render_qc_logits [v, q, c+1, h, w] * b list
        for idx, (render_qc_logit, q_score) in enumerate(
            zip(render_qc_logits, context_seg_query_scores)
        ):
            # render_qc_logit [v, q, c+1, h, w]
            v, q, c, h, w = render_qc_logit.shape
            c_logit, q_index = render_qc_logit.max(dim=1)
            # c_logit [v, c+1, h, w]， q_index [v, c+1, h, w]
            c_logit = torch.concat(
                [c_logit[:, -1:, :, :], c_logit[:, :-1, :, :]], dim=1
            )
            q_index = torch.concat(
                [q_index[:, -1:, :, :], q_index[:, :-1, :, :]], dim=1
            )
            sem_logits, sem_id = c_logit.max(dim=1)  # [v, h, w]
            vidx, hidx, widx = torch.meshgrid(
                torch.arange(v),
                torch.arange(h),
                torch.arange(w),
                indexing="ij",
            )
            vidx = vidx.to(self.device)
            hidx = hidx.to(self.device)
            widx = widx.to(self.device)
            q_index = q_index[vidx, sem_id, hidx, widx] + 1  # [v, h, w]
            sem_masks = sem_logits < 0.3
            sem_id[sem_masks] = 0
            q_index[sem_id == 0] = 0
            info = []
            for q_idx, q_score_i in enumerate(q_score):
                q_sem_ids = sem_id[q_index == q_idx + 1]
                if q_sem_ids.numel() == 0:
                    continue
                else:
                    q_sem_id = q_sem_ids[0]
                info.append(
                    {
                        "id": q_idx + 1,
                        "label_id": q_sem_id.item(),
                        "was_fused": False,
                        "score": q_score_i,
                    }
                )

            for stuff in self.pipecfg.model.mask2former.label_ids_to_fuse:
                stuff_mask = sem_id == (stuff + 1)
                q_index[stuff_mask] = (
                    self.pipecfg.model.mask2former.num_queries + stuff + 1
                )
                for i in info:
                    if i["label_id"] == (stuff + 1):
                        i["was_fused"] = True
                        i["id"] = q_index[stuff_mask][0].item()
            all_sem_id.append(sem_id)
            all_ins_id.append(q_index)
            seg_infos.append(info)
        all_sem_id = torch.stack(all_sem_id, dim=0)  # [b, v, h, w]
        all_ins_id = torch.stack(all_ins_id, dim=0)  # [b, v, h, w]
        context_sem_ids = []
        context_ins_ids = []
        for i in range(len(context_in_target_ids)):
            context_sem_ids.append(all_sem_id[i, context_in_target_ids[i], :, :])
            context_ins_ids.append(all_ins_id[i, context_in_target_ids[i], :, :])
        context_sem_ids = torch.stack(context_sem_ids, dim=0)  # [b, h, w]
        context_ins_ids = torch.stack(context_ins_ids, dim=0)  # [b, h, w]
        target_sem_ids = all_sem_id
        target_ins_ids = all_ins_id
        return (
            gaussians,
            render_output,
            context_seg_output,
            context_seg_masks,
            context_seg_infos,
            context_sem_ids,
            context_ins_ids,
            target_sem_ids,
            target_ins_ids,
            seg_infos,
        )

    def training_step(self, batch, batch_idx):
        global_step = self.global_step
        loss = 0
        target_views_images = batch["target_views_images"]

        context_ids = batch["context_views_id"]
        target_ids = batch["target_views_id"]
        context_in_target_ids = []
        for context_id, target_id in zip(context_ids, target_ids):
            context_in_target_id = []
            for pos, idx in enumerate(target_id):
                if idx in context_id:
                    context_in_target_id.append(pos)
            context_in_target_ids.append(context_in_target_id)

        (
            gaussians,
            context_seg_output,
            context_seg_masks,
            context_seg_infos,
            render_output,
        ) = self.step_wo_lift(batch, batch_idx)
        context_seg_loss = context_seg_output.loss.squeeze()
        self.log("train/context_seg_loss", context_seg_loss, prog_bar=True)
        loss += self.pipecfg.weight_seg_loss * context_seg_loss

        context_render_depth = []
        for i in range(len(context_in_target_ids)):
            context_render_depth.append(
                render_output["render_depth"][i, context_in_target_ids[i], ...]
            )
        context_render_depth = torch.stack(context_render_depth, dim=0)  # [B, N, H, W]
        seg_mask = torch.stack(context_seg_masks, dim=0)
        depth_smoothness_loss = 0
        depth_dx = context_render_depth.diff(dim=-1)
        depth_dy = context_render_depth.diff(dim=-2)
        instance_dx = ~(seg_mask.diff(dim=-1).to(torch.bool))
        instance_dx[seg_mask[:, :, :, 1:] == -1] = False
        instance_dy = ~(seg_mask.diff(dim=-2).to(torch.bool))
        instance_dy[seg_mask[:, :, 1:, :] == -1] = False
        depth_dx = depth_dx * instance_dx.detach()
        depth_dy = depth_dy * instance_dy.detach()
        depth_smoothness_loss += depth_dx.abs().mean()
        depth_smoothness_loss += depth_dy.abs().mean()
        loss += self.pipecfg.weight_depth_smoothness * depth_smoothness_loss
        self.log(
            "train/depth_smoothness_loss",
            depth_smoothness_loss,
            prog_bar=True,
        )
        loss += self.calc_render_loss(
            render_output=render_output,
            target_views_images=target_views_images,
        )
        self.log("train/loss", loss, prog_bar=True)
        if global_step % self.pipecfg.log_training_result_interval == 0:
            self.visualizer.add(
                save_dir=self.get_log_dir(),
                batch=batch,
                render_output=render_output,
                gaussians=gaussians,
                context_seg_masks=context_seg_masks,
                context_seg_infos=context_seg_infos,
            )
            self.visualizer.write_files()
        return loss

    def on_validation_epoch_start(self):
        self.evaluator = Evaluator(self.pipecfg.evaluator)
        self.visualizer.reset()
        if self.global_rank == 0:
            self.evaluator.setup()

    def validation_step(self, batch, batch_idx):
        (
            gaussians,
            render_output,
            context_seg_output,
            context_seg_masks,
            context_seg_infos,
            context_sem_ids,
            context_ins_ids,
            target_sem_ids,
            target_ins_ids,
            seg_infos,
        ) = self.step_w_query_class_logit_lift(batch, batch_idx)
        self.visualizer.add(
            save_dir=self.get_log_dir(),
            batch=batch,
            render_output=render_output,
            gaussians=gaussians,
            context_semantic_ids=context_sem_ids,
            context_instance_ids=context_ins_ids,
            target_semantic_ids=target_sem_ids,
            target_instance_ids=target_ins_ids,
            context_seg_infos=seg_infos,
            target_seg_infos=seg_infos,
        )

    def on_validation_epoch_end(self):
        self.visualizer.write_files()
        self.trainer.strategy.barrier()
        if self.global_rank == 0:
            result = self.evaluator.evaluate(self.get_log_dir())
            for key, value in result.items():
                if isinstance(value, float):
                    self.log(f"val/{key}", value)
                elif isinstance(value, dict) and "map" in key:
                    self.log(f"val/{key}", value["map"])
            del self.evaluator
        self.trainer.strategy.barrier()

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def calc_render_loss(
        self,
        render_output,
        target_views_images,
    ):
        B, N, C, H, W = target_views_images.shape
        loss = 0
        render_colors: Tensor = render_output["render_color"]  # [B, N, 3, H, W]
        render_loss = self.mse_loss(render_colors, target_views_images)
        loss += render_loss
        self.log("train/render_loss", render_loss, prog_bar=True)
        lpips_loss = self.lpips(
            F.interpolate(
                render_colors.view(B * N, C, H, W),
                size=(H // 2, W // 2),
                mode="bilinear",
                align_corners=True,
            ),
            F.interpolate(
                target_views_images.view(B * N, C, H, W),
                size=(H // 2, W // 2),
                mode="bilinear",
                align_corners=True,
            ),
        )
        self.log("train/lpips_loss", lpips_loss, prog_bar=True)
        loss += 0.5 * lpips_loss
        return loss

    def configure_optimizers(self):
        lr = self.cfg.optimizer.lr
        warm_up_epochs = self.cfg.optimizer.warm_up_epochs
        normal_lr_params, normal_lr_param_names = [], []
        high_lr_params, high_lr_param_names = [], []
        low_lr_params, low_lr_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "gaussian_param_head" in name or "intrinsic_encoder" in name:
                normal_lr_params.append(param)
                normal_lr_param_names.append(name)
            elif "mask2former" in name or "adapter" in name:
                high_lr_params.append(param)
                high_lr_param_names.append(name)
            else:
                low_lr_params.append(param)
                low_lr_param_names.append(name)

        param_dicts = [
            {
                "params": normal_lr_params,
                "lr": 5 * lr,
            },
            {
                "params": high_lr_params,
                "lr": 10 * lr,
            },
            {
                "params": low_lr_params,
                "lr": lr * 0.1,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=lr, weight_decay=0.05, betas=(0.9, 0.95)
        )
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_epochs,
            1,
            total_iters=warm_up_epochs,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - warm_up_epochs,
            eta_min=lr * 0.05,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }

    def get_log_dir(self) -> str:
        stage_map = {
            "sanity_check": "sanity_check",
            "train": "train",
            "validate": "val",
            "test": "test",
            "predict": "pred",
        }
        log_dir = f"{self.cfg.output_path}/{stage_map[self.trainer.state.stage]}/{self.trainer.global_step}"
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
