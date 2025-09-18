import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import os
from src.models.backbone_croco import AsymmetricCroCoMulti
from src.models.vit_adapter import CroCoViTAdapter
from src.models.croco.misc import transpose_to_landscape
from src.models.heads import head_factory
from src.models.gaussian_adapter import UnifiedGaussianAdapter
from transformers import Mask2FormerConfig
from src.models.mask2former import VideoMask2FormerForVideoSegmentation
from src.models.mask2former.image_processing_video_mask2former import (
    VideoMask2FormerImageProcessor,
)
from einops import rearrange

from src.utils.weight_modify import checkpoint_filter_fn
from src.utils.gaussians_types import Gaussians

from src.config import ModelCfg

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class SIU3RMultiViewModel(nn.Module):
    def __init__(
        self,
        cfg: ModelCfg,
    ):
        super(SIU3RMultiViewModel, self).__init__()
        self.cfg = cfg
        self._set_backbone()
        self._set_adapter()
        self._set_mask2former()
        self._set_center_head()
        self._set_gaussian_param_head()
        self._set_gaussian_adapter()

    def _set_backbone(self):
        self.backbone = AsymmetricCroCoMulti(
            img_size=self.cfg.image_size, **self.cfg.croco.__dict__
        )
        self.backbone.depth_mode = ("exp", -float("inf"), float("inf"))
        self.backbone.conf_mode = None
        if self.cfg.croco.freeze == "encoder":
            self.backbone.enc_blocks.eval()
            for param in self.backbone.enc_blocks.parameters():
                param.requires_grad = False
            self.backbone.enc_norm.eval()
            for param in self.backbone.enc_norm.parameters():
                param.requires_grad = False
            self.backbone.patch_embed.eval()
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False

    def _set_adapter(self):
        self.adapter = CroCoViTAdapter(
            num_block=self.cfg.croco.enc_depth,
            embed_dim=self.cfg.croco.enc_embed_dim,
            size=self.cfg.image_size,
            patchsize=self.cfg.croco.patch_size,
        )

    def _set_mask2former(self):
        self.mask2former_config = Mask2FormerConfig(
            id2label=self.cfg.mask2former.id2label,
            num_queries=self.cfg.mask2former.num_queries,
            train_refer_segmentation=False,
        )
        self.mask2former = VideoMask2FormerForVideoSegmentation(self.mask2former_config)
        self.mask2former.train()
        self.processor = VideoMask2FormerImageProcessor()

    def _set_center_head(self):
        self.downstream_head1 = head_factory(
            "dpt", "pts3d", self.backbone, has_conf=False
        )
        self.downstream_head2 = head_factory(
            "dpt", "pts3d", self.backbone, has_conf=False
        )
        self.head1 = transpose_to_landscape(self.downstream_head1)
        self.head2 = transpose_to_landscape(self.downstream_head2)

    def _set_gaussian_param_head(self):
        # sh + 3 scale + 4 rotation + 1 opacity
        self.raw_gs_dim = (self.cfg.gaussian_head.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1
        self.gaussian_param_head1 = head_factory(
            "dpt_gs",
            "gs_params",
            self.backbone,
            has_conf=False,
            out_nchan=self.raw_gs_dim,
        )
        self.gaussian_param_head2 = head_factory(
            "dpt_gs",
            "gs_params",
            self.backbone,
            has_conf=False,
            out_nchan=self.raw_gs_dim,
        )

    def _set_gaussian_adapter(self):
        self.gaussian_adapter = UnifiedGaussianAdapter(
            self.cfg.gaussian_head.gaussian_scale_min,
            self.cfg.gaussian_head.gaussian_scale_max,
            sh_degree=self.cfg.gaussian_head.sh_degree,
        )

    def load_recon_ckpt(self):
        recon_ckpt_path = os.path.join(
            self.cfg.pretrained_weights_path,
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        )
        recon_ckpt = torch.load(recon_ckpt_path, map_location="cpu", weights_only=False)
        recon_ckpt = recon_ckpt["model"]
        recon_ckpt = checkpoint_filter_fn(recon_ckpt, self)
        missing_keys, unexpected_keys = self.load_state_dict(recon_ckpt, strict=False)
        log.info(f"loaded recon ckpt from {recon_ckpt_path}")

    def load_dust3r_ckpt(self):
        recon_ckpt_path = os.path.join(
            self.cfg.pretrained_weights_path,
            "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        )
        recon_ckpt = torch.load(recon_ckpt_path, map_location="cpu", weights_only=False)
        recon_ckpt = recon_ckpt["model"]
        recon_ckpt = checkpoint_filter_fn(recon_ckpt, self)
        missing_keys, unexpected_keys = self.load_state_dict(recon_ckpt, strict=False)
        log.info(f"loaded recon ckpt from {recon_ckpt_path}")

    def load_seg_ckpt(self):
        seg_ckpt_path = os.path.join(
            self.cfg.pretrained_weights_path,
            "panoptic_coco_pretrain_vitadapter_maskdecoder_epoch60.ckpt",
        )
        seg_ckpt = torch.load(seg_ckpt_path, map_location="cpu", weights_only=False)
        seg_ckpt = seg_ckpt["state_dict"]
        for key in list(seg_ckpt.keys()):
            if "class_predictor" in key or "criterion" in key or "backbone" in key:
                seg_ckpt.pop(key)
            elif "queries_embedder" in key or "queries_features" in key:
                pretrained_queries = seg_ckpt.pop(key)
                num_q, q_dim = pretrained_queries.shape
                tmp_queries = nn.Embedding(
                    self.cfg.mask2former.num_queries,
                    q_dim,
                    device=pretrained_queries.device,
                )
                tmp_queries.weight.data[:num_q] = pretrained_queries
                seg_ckpt[key[len("model.") :]] = tmp_queries.weight.data
            else:
                seg_ckpt[key[len("model.") :]] = seg_ckpt.pop(key)
        seg_missing_keys, seg_unexpected_keys = self.load_state_dict(
            seg_ckpt, strict=False
        )
        log.info(f"loaded ckpt from {seg_ckpt_path}")
        # log.info(f"missing_keys: {intersaction_missing_keys}")
        # log.info(f"unexpected_keys: {union_unexpected_keys}")

    def delete_recon_part(self):
        del self.backbone.dec_blocks
        del self.backbone.dec_blocks2
        del self.backbone.decoder_embed
        del self.downstream_head1
        del self.downstream_head2
        del self.head1
        del self.head2
        del self.gaussian_param_head1
        del self.gaussian_param_head2
        del self.gaussian_adapter

    def gaussian_center(self, decs, shapes):
        all_centers = []
        dec1 = decs[0]
        dec_others = decs[1:]
        res1 = self.head1([tok.float() for tok in dec1], shapes[0])
        all_centers.append(res1)
        for dec2, shape2 in zip(dec_others, shapes[1:]):
            res2 = self.head2([tok.float() for tok in dec2], shape2)
            all_centers.append(res2)
        return all_centers

    def gaussian_param(self, decs, views, shapes):
        views = views.permute(1, 0, 2, 3, 4)  # [B, V, C, H, W] -> [V, B, C, H, W]
        all_GS_res = []
        GS_res1 = self.gaussian_param_head1(
            [tok.float() for tok in decs[0]],
            None,
            views[0],
            shapes[0][0].cpu().tolist(),
        )
        GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
        all_GS_res.append(GS_res1)

        for dec2, view2, shape2 in zip(decs[1:], views[1:], shapes[1:]):
            GS_res2 = self.gaussian_param_head2(
                [tok.float() for tok in dec2],
                None,
                view2,
                shape2[0].cpu().tolist(),
            )
            GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
            all_GS_res.append(GS_res2)
        return all_GS_res

    def gaussian(self, decs, shapes, views):
        shapes = shapes.permute(1, 0, 2)  # [B, V, 2] -> [V, B, 2]
        all_res = self.gaussian_center(decs, shapes)
        all_GS_res = self.gaussian_param(decs, views, shapes)

        pts_all = torch.stack(
            [res["pts3d"] for res in all_res], dim=1
        )  # [B, N, H, W, D]
        pts_all = rearrange(pts_all, "b n h w d -> b n (h w) d")
        gaussians = torch.stack(all_GS_res, dim=1)  # [B, N, HW, D]
        gaussians = self.gaussian_adapter.forward(
            pts_all,  # [B, N, HW, D]
            gaussians,
        )

        return gaussians, pts_all

    def post_process_gaussians(
        self,
        B,
        V,
        H,
        W,
        gaussians: Gaussians,
        context_seg_output,
        enable_query_class_logit_lift=False,
    ):
        context_seg_results = self.processor.post_process_panoptic_segmentation(
            outputs=context_seg_output,
            threshold=self.cfg.mask2former.seg_threshold,
            target_sizes=[(H, W)] * B,
            label_ids_to_fuse=set(self.cfg.mask2former.label_ids_to_fuse),
        )

        context_seg_masks = []
        context_seg_infos = []
        if enable_query_class_logit_lift:
            context_seg_query_class_logits = []
            context_seg_query_scores = []
        for context_seg_result in context_seg_results:
            seg_mask = context_seg_result["segmentation"]
            context_seg_masks.append(seg_mask)
            seg_info = context_seg_result["segments_info"]
            context_seg_infos.append(seg_info)
            if enable_query_class_logit_lift:
                seg_query_class_logits = context_seg_result[
                    "query_class_logits"
                ]  # [n, q, c+1, h, w]
                seg_query_class_logits = rearrange(
                    seg_query_class_logits, "n q c h w -> (n h w) q c "
                )
                context_seg_query_class_logits.append(seg_query_class_logits)
                context_seg_query_scores.append(context_seg_result["query_scores"])

        gaussians.semantic_labels = torch.zeros(
            B, V, H, W, dtype=torch.int32, device=gaussians.means.device
        )
        gaussians.instance_labels = torch.zeros(
            B, V, H, W, dtype=torch.int32, device=gaussians.means.device
        )
        if enable_query_class_logit_lift:
            gaussians.seg_query_class_logits = context_seg_query_class_logits

        for b, (info, masks) in enumerate(
            zip(context_seg_infos, context_seg_masks)
        ):  # iterate over batch
            if len(info) == 0:
                continue
            for seg in info:  # iterate over segments
                semantic_label = (
                    seg["label_id"] + 1
                )  # note here + 1 for 0 is background
                instance_label = seg["id"]
                mask = masks == seg["id"]
                gaussians.semantic_labels[b, mask] = semantic_label
                gaussians.instance_labels[b, mask] = instance_label
        gaussians.semantic_labels = rearrange(
            gaussians.semantic_labels, "b n h w -> b (n h w)"
        )
        gaussians.instance_labels = rearrange(
            gaussians.instance_labels, "b n h w -> b (n h w)"
        )
        gaussians.means = rearrange(gaussians.means, "b n r d -> b (n r) d")
        gaussians.covariances = rearrange(
            gaussians.covariances, "b n r i j -> b (n r) i j"
        )
        gaussians.scales = rearrange(gaussians.scales, "b n r d -> b (n r) d")
        gaussians.rotations = rearrange(gaussians.rotations, "b n r d -> b (n r) d")
        gaussians.opacities = rearrange(gaussians.opacities, "b n r -> b (n r)")
        gaussians.harmonics = rearrange(
            gaussians.harmonics, "b n r sh d -> b (n r) sh d"
        )

        return (
            gaussians,
            context_seg_output,
            context_seg_masks,
            context_seg_infos,
            (context_seg_query_scores if enable_query_class_logit_lift else None),
        )

    def forward(
        self,
        context_views_images,
        context_views_intrinsics,
        mask_labels=None,
        class_labels=None,
        enable_query_class_logit_lift=False,
    ):
        B, V, _, H, W = context_views_images.shape
        croco_input = {
            "image": context_views_images,
            "intrinsics": context_views_intrinsics,
            "near": 0.1,
            "far": 100,
        }
        (feat, all_feat, dec_feat, shape, images) = self.backbone.forward(
            croco_input, return_views=True
        )
        multi_scale_feats = []
        all_feats = []
        dec_feats = []
        for vid in range(V):
            img = context_views_images[:, vid]
            all_feat_vid = []
            all_dec_feat_vid = []
            for feat_i in all_feat:
                all_feat_vid.append(feat_i[:, vid])
            for dec_feat_i in dec_feat:
                all_dec_feat_vid.append(dec_feat_i[:, vid])
            dec_feats.append(all_dec_feat_vid)
            all_feats.append(all_feat_vid)
            multi_scale_feat = self.adapter(img, all_feat_vid)
            multi_scale_feats.append(multi_scale_feat)

        gaussians, all_pts = self.gaussian(
            dec_feats,
            shape,
            images,
        )

        multi_scale_feat_aggregate = []
        for i in range(len(multi_scale_feats[0])):
            multi_scale_feat_i = []
            for vid in range(V):
                multi_scale_feat_i.append(multi_scale_feats[vid][i])
            multi_scale_feat_aggregate.append(torch.stack(multi_scale_feat_i, dim=1))

        context_seg_output = self.mask2former(
            multi_scale_feat=multi_scale_feat_aggregate,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        (
            gaussians,
            context_seg_output,
            context_seg_masks,
            context_seg_infos,
            context_seg_query_scores,
        ) = self.post_process_gaussians(
            B,
            V,
            H,
            W,
            gaussians,
            context_seg_output,
            enable_query_class_logit_lift=enable_query_class_logit_lift,
        )
        if enable_query_class_logit_lift:
            return (
                gaussians,
                context_seg_output,
                context_seg_masks,
                context_seg_infos,
                context_seg_query_scores,
            )
        else:
            return (
                gaussians,
                context_seg_output,
                context_seg_masks,
                context_seg_infos,
            )

    def seg_forward(
        self,
        context_views_images,
        context_views_intrinsics,
        mask_labels=None,
        class_labels=None,
        enable_query_class_logit_lift=False,
    ):
        B, V, _, H, W = context_views_images.shape
        croco_input = {
            "image": context_views_images,
            "intrinsics": context_views_intrinsics,
            "near": 0.1,
            "far": 100,
        }
        (feat, all_feat, dec_feat, shape, images) = self.backbone.forward(
            croco_input, return_views=True
        )
        multi_scale_feats = []
        all_feats = []
        for vid in range(V):
            img = context_views_images[:, vid]
            all_feat_vid = []
            for feat_i in all_feat:
                all_feat_vid.append(feat_i[:, vid])
            all_feats.append(all_feat_vid)
            multi_scale_feat = self.adapter(img, all_feat_vid)
            multi_scale_feats.append(multi_scale_feat)

        multi_scale_feat_aggregate = []
        for i in range(len(multi_scale_feats[0])):
            multi_scale_feat_i = []
            for vid in range(V):
                multi_scale_feat_i.append(multi_scale_feats[vid][i])
            multi_scale_feat_aggregate.append(torch.stack(multi_scale_feat_i, dim=1))

        context_seg_output = self.mask2former(
            multi_scale_feat=multi_scale_feat_aggregate,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        context_seg_results = self.processor.post_process_panoptic_segmentation(
            outputs=context_seg_output,
            threshold=self.cfg.mask2former.seg_threshold,
            target_sizes=[(H, W)] * B,
            label_ids_to_fuse=set(self.cfg.mask2former.label_ids_to_fuse),
        )

        context_seg_masks = []
        context_seg_infos = []
        if enable_query_class_logit_lift:
            context_seg_query_class_logits = []
            context_seg_query_scores = []
        for context_seg_result in context_seg_results:
            seg_mask = context_seg_result["segmentation"]
            context_seg_masks.append(seg_mask)
            seg_info = context_seg_result["segments_info"]
            context_seg_infos.append(seg_info)
            if enable_query_class_logit_lift:
                seg_query_class_logits = context_seg_result[
                    "query_class_logits"
                ]  # [n, q, c+1, h, w]
                seg_query_class_logits = rearrange(
                    seg_query_class_logits, "n q c h w -> (n h w) q c "
                )
                context_seg_query_class_logits.append(seg_query_class_logits)
                context_seg_query_scores.append(context_seg_result["query_scores"])

        return (
            context_seg_output,
            context_seg_masks,
            context_seg_infos,
        )
