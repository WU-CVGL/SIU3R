# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn

# import dust3r.utils.path_to_croco
from .dpt_block import DPTOutputAdapter, Interpolate, make_fusion_block
from .head_modules import UnetExtractor
from .postprocess import postprocess


class MultiResolutionDPTOutputAdapter(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    and output multi-resolution features
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

        self.input_merger_ds4 = nn.Sequential(
            Interpolate(scale_factor=0.25, mode="bilinear", align_corners=True),
            nn.Conv2d(3, 256, 7, 1, 3),
            nn.ReLU(),
        )
        self.input_merger_ds8 = nn.Sequential(
            Interpolate(scale_factor=0.125, mode="bilinear", align_corners=True),
            nn.Conv2d(3, 256, 7, 1, 3),
            nn.ReLU(),
        )
        self.input_merger_ds16 = nn.Sequential(
            Interpolate(scale_factor=0.0625, mode="bilinear", align_corners=True),
            nn.Conv2d(3, 256, 7, 1, 3),
            nn.ReLU(),
        )
        self.input_merger_ds32 = nn.Sequential(
            Interpolate(scale_factor=0.03125, mode="bilinear", align_corners=True),
            nn.Conv2d(3, 256, 7, 1, 3),
            nn.ReLU(),
        )
        self.scratch.refinenet1 = make_fusion_block(
            self.feature_dim, False, 1, skip_upsample=True
        )
        self.scratch.refinenet2 = make_fusion_block(
            self.feature_dim, False, 1, skip_upsample=True
        )
        self.scratch.refinenet3 = make_fusion_block(
            self.feature_dim, False, 1, skip_upsample=True
        )
        self.scratch.refinenet4 = make_fusion_block(
            self.feature_dim, False, 1, skip_upsample=True
        )
        del self.head
        self.head_ds4 = nn.Sequential(
            nn.Conv2d(
                self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(self.feature_dim, self.num_channels, kernel_size=1),
        )
        self.head_ds8 = nn.Sequential(
            nn.Conv2d(
                self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(self.feature_dim, self.num_channels, kernel_size=1),
        )
        self.head_ds16 = nn.Sequential(
            nn.Conv2d(
                self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(self.feature_dim, self.num_channels, kernel_size=1),
        )
        self.head_ds32 = nn.Sequential(
            nn.Conv2d(
                self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(self.feature_dim, self.num_channels, kernel_size=1),
        )

    def forward(
        self,
        encoder_tokens: List[torch.Tensor],
        depths,
        imgs,
        image_size=None,
        conf=None,
    ):
        assert (
            self.dim_tokens_enc is not None
        ), "Need to call init(dim_tokens_enc) function first"
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [
            rearrange(l, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W) for l in layers
        ]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])  # downsampled to 1/32
        up_path4 = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)(
            path_4
        )[:, :, : layers[2].shape[2], : layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(up_path4, layers[2])  # downsampled to 1/16
        up_path3 = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)(
            path_3
        )
        path_2 = self.scratch.refinenet2(up_path3, layers[1])  # downsampled to 1/8
        up_path2 = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)(
            path_2
        )
        path_1 = self.scratch.refinenet1(up_path2, layers[0])  # downsampled to 1/4

        direct_img_feat_ds4 = self.input_merger_ds4(imgs)
        path_1 = path_1 + direct_img_feat_ds4
        out_ds4 = self.head_ds4(path_1)
        direct_img_feat_ds8 = self.input_merger_ds8(imgs)
        path_2 = path_2 + direct_img_feat_ds8
        out_ds8 = self.head_ds8(path_2)
        direct_img_feat_ds16 = self.input_merger_ds16(imgs)
        path_3 = path_3 + direct_img_feat_ds16
        out_ds16 = self.head_ds16(path_3)
        direct_img_feat_ds32 = self.input_merger_ds32(imgs)
        path_4 = path_4 + direct_img_feat_ds32
        out_ds32 = self.head_ds32(path_4)
        return [out_ds4, out_ds8, out_ds16, out_ds32]


class PixelwiseTaskWithDPT(nn.Module):
    """DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(
        self,
        *,
        n_cls_token=0,
        hooks_idx=None,
        dim_tokens=None,
        output_width_ratio=1,
        num_channels=1,
        postprocess=None,
        depth_mode=None,
        conf_mode=None,
        **kwargs
    ):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(
            output_width_ratio=output_width_ratio, num_channels=num_channels, **kwargs
        )
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = MultiResolutionDPTOutputAdapter(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, depths, imgs, img_info, conf=None):
        out = self.dpt(
            x, depths, imgs, image_size=(img_info[0], img_info[1]), conf=conf
        )
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_multi_res_gs_dpt_head(
    net,
    has_conf=False,
    out_nchan=3,
    postprocess_func=postprocess,
):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    assert (
        postprocess_func is None
    ), "postprocess_func should be None for multi_res_dpt_gs_head"
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim // 2
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(
        num_channels=out_nchan + has_conf,
        feature_dim=feature_dim,
        last_dim=last_dim,
        hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
        dim_tokens=[ed, dd, dd, dd],
        postprocess=postprocess_func,
        depth_mode=net.depth_mode,
        conf_mode=net.conf_mode,
        head_type="gs_params",
    )
