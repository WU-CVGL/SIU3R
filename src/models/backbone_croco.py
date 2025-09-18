from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from einops import rearrange
from .croco.blocks import DecoderBlock
from .croco.croco import CroCoNet
from .croco.misc import (
    freeze_all_params,
    transpose_to_landscape,
    is_symmetrized,
    interleave,
    make_batch_symmetric,
)
from .croco.patch_embed import PatchEmbedDust3R

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class AsymmetricCroCo(CroCoNet):
    """Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(
        self,
        enc_depth=24,
        dec_depth=12,
        enc_embed_dim=1024,
        dec_embed_dim=768,
        enc_num_heads=16,
        dec_num_heads=12,
        pos_embed="RoPE100",
        img_size=(480, 640),  # note that this is ignored, but kept for compatibility
        patch_size=16,
        freeze="none",
        **kwargs,
    ) -> None:
        super().__init__(
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            enc_embed_dim=enc_embed_dim,
            dec_embed_dim=dec_embed_dim,
            enc_num_heads=enc_num_heads,
            dec_num_heads=dec_num_heads,
            pos_embed=pos_embed,
            img_size=img_size,
            patch_size=patch_size,
        )

        self.dec_blocks2 = deepcopy(
            self.dec_blocks
        )  # This is used in DUSt3R and MASt3R
        self.intrinsic_encoder = nn.Linear(9, enc_embed_dim)  # 3x3 intrinsics matrix

        self.set_freeze(freeze)

    def _set_patch_embed(
        self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3
    ):
        # note there img_size will be ignored
        self.patch_embed = PatchEmbedDust3R(
            img_size,
            patch_size,
            in_chans,
            enc_embed_dim,
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks2") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks2")] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ["none", "mask", "encoder"], f"unexpected freeze={freeze}"
        to_be_frozen = {
            "none": [],
            "encoder": [self.patch_embed, self.enc_blocks],
            "encoder_decoder": [
                self.patch_embed,
                self.enc_blocks,
                self.enc_norm,
                self.decoder_embed,
                self.dec_blocks,
                self.dec_blocks2,
                self.dec_norm,
            ],
        }
        log.info(f"Freezing {freeze}")
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        pass

    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = None

    def _set_mask_generator(self, num_patches, mask_ratio):
        """No mask generator"""
        pass

    def _encode_image(self, image, true_shape, intrinsics_embed=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        x = torch.cat((x, intrinsics_embed), dim=1)
        add_pose = pos[:, 0:1, :].clone()
        add_pose[:, :, 0] += pos[:, -1, 0].unsqueeze(-1) + 1
        pos = torch.cat((pos, add_pose), dim=1)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        out = []
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)
            out.append(x)

        x = self.enc_norm(x)
        return x, pos, None, out

    def _encode_image_pairs(
        self,
        img1,
        img2,
        true_shape1,
        true_shape2,
        intrinsics_embed1=None,
        intrinsics_embed2=None,
    ):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _, all_feat = self._encode_image(
                torch.cat((img1, img2), dim=0),
                torch.cat((true_shape1, true_shape2), dim=0),
                (
                    torch.cat((intrinsics_embed1, intrinsics_embed2), dim=0)
                    if intrinsics_embed1 is not None
                    else None
                ),
            )
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
            all_feat1 = []
            all_feat2 = []
            for feat in all_feat:
                feat1, feat2 = feat.chunk(2, dim=0)
                all_feat1.append(feat1)
                all_feat2.append(feat2)
        else:
            out, pos, _, all_feat1 = self._encode_image(
                img1, true_shape1, intrinsics_embed1
            )
            out2, pos2, _, all_feat2 = self._encode_image(
                img2, true_shape2, intrinsics_embed2
            )
        return out, out2, pos, pos2, all_feat1, all_feat2

    def _encode_symmetrized(self, view1, view2, force_asym=False):
        img1 = view1["img"]
        img2 = view2["img"]
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get(
            "true_shape", torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        )
        shape2 = view2.get(
            "true_shape", torch.tensor(img2.shape[-2:])[None].repeat(B, 1)
        )
        # warning! maybe the images have different portrait/landscape orientations
        intrinsics_embed1 = view1.get("intrinsics_embed", None)
        intrinsics_embed2 = view2.get("intrinsics_embed", None)

        if force_asym or not is_symmetrized(view1, view2):
            feat1, feat2, pos1, pos2, all_feat1, all_feat2 = self._encode_image_pairs(
                img1, img2, shape1, shape2, intrinsics_embed1, intrinsics_embed2
            )
        else:
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2, all_feat1, all_feat2 = self._encode_image_pairs(
                img1[::2], img2[::2], shape1[::2], shape2[::2]
            )
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
            all_feat1 = map(interleave, all_feat1)
            all_feat2 = map(interleave, all_feat2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2), (all_feat1, all_feat2)

    def _decoder(self, f1, pos1, f2, pos2, extra_embed1=None, extra_embed2=None):
        final_output = [(f1, f2)]  # before projection

        if extra_embed1 is not None:
            f1 = torch.cat((f1, extra_embed1), dim=-1)
        if extra_embed2 is not None:
            f2 = torch.cat((f2, extra_embed2), dim=-1)

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f"head{head_num}")
        return head(decout, img_shape)

    def forward(
        self,
        context: dict,
        symmetrize_batch=False,
        return_views=False,
    ):
        b, v, _, h, w = context["image"].shape
        device = context["image"].device

        view1, view2 = (
            {"img": context["image"][:, 0]},
            {"img": context["image"][:, 1]},
        )

        # camera embedding in the encoder
        intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2))
        view1["intrinsics_embed"] = intrinsic_embedding[:, 0].unsqueeze(1)
        view2["intrinsics_embed"] = intrinsic_embedding[:, 1].unsqueeze(1)

        if symmetrize_batch:
            instance_list_view1, instance_list_view2 = (
                [0 for _ in range(b)],
                [1 for _ in range(b)],
            )
            view1["instance"] = instance_list_view1
            view2["instance"] = instance_list_view2
            view1["idx"] = instance_list_view1
            view2["idx"] = instance_list_view2
            view1, view2 = make_batch_symmetric(view1, view2)

            # encode the two images --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2), (all_feat1, all_feat2) = (
                self._encode_symmetrized(view1, view2, force_asym=False)
            )
        else:
            # encode the two images --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2), (all_feat1, all_feat2) = (
                self._encode_symmetrized(view1, view2, force_asym=True)
            )
        # check if self has attribute "dec_blocks"
        if hasattr(self, "dec_blocks"):
            dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
            dec1, dec2 = list(dec1), list(dec2)
            for i in range(len(dec1)):
                dec1[i] = dec1[i][:, :-1]
                dec2[i] = dec2[i][:, :-1]
        else:
            dec1, dec2 = None, None
        feat1 = feat1[:, :-1]
        feat2 = feat2[:, :-1]
        for i in range(len(all_feat1)):
            all_feat1[i] = all_feat1[i][:, :-1]
            all_feat2[i] = all_feat2[i][:, :-1]

        if return_views:
            return (
                feat1,
                feat2,
                all_feat1,
                all_feat2,
                dec1,
                dec2,
                shape1,
                shape2,
                view1,
                view2,
            )
        return (
            feat1,
            feat2,
            all_feat1,
            all_feat2,
            dec1,
            dec2,
            shape1,
            shape2,
        )

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024


class AsymmetricCroCoMulti(CroCoNet):
    """Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(
        self,
        enc_depth=24,
        dec_depth=12,
        enc_embed_dim=1024,
        dec_embed_dim=768,
        enc_num_heads=16,
        dec_num_heads=12,
        pos_embed="RoPE100",
        img_size=(480, 640),  # note that this is ignored, but kept for compatibility
        patch_size=16,
        freeze="none",
        **kwargs,
    ) -> None:
        super().__init__(
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            enc_embed_dim=enc_embed_dim,
            dec_embed_dim=dec_embed_dim,
            enc_num_heads=enc_num_heads,
            dec_num_heads=dec_num_heads,
            pos_embed=pos_embed,
            img_size=img_size,
            patch_size=patch_size,
        )

        self.dec_blocks2 = deepcopy(
            self.dec_blocks
        )  # This is used in DUSt3R and MASt3R
        self.intrinsic_encoder = nn.Linear(9, enc_embed_dim)

        # self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

    def _set_patch_embed(
        self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3
    ):
        # note there img_size will be ignored
        self.patch_embed = PatchEmbedDust3R(
            img_size,
            patch_size,
            in_chans,
            enc_embed_dim,
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks2") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks2")] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ["none", "mask", "encoder"], f"unexpected freeze={freeze}"
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token],
            "encoder": [self.mask_token, self.patch_embed, self.enc_blocks],
            "encoder_decoder": [
                self.mask_token,
                self.patch_embed,
                self.enc_blocks,
                self.enc_norm,
                self.decoder_embed,
                self.dec_blocks,
                self.dec_blocks2,
                self.dec_norm,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def _encode_image(self, image, true_shape, intrinsics_embed=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        if intrinsics_embed is not None:
            x = torch.cat((x, intrinsics_embed), dim=1)
            add_pose = pos[:, 0:1, :].clone()
            add_pose[:, :, 0] += pos[:, -1, 0].unsqueeze(-1) + 1
            pos = torch.cat((pos, add_pose), dim=1).contiguous()

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        out = []
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)
            out.append(x)

        x = self.enc_norm(x)
        return x, pos, None, out

    def _decoder(self, feat, pose, extra_embed=None):
        b, v, l, c = feat.shape
        final_output = [feat]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c")
        f = self.decoder_embed(f)
        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v)
        final_output.append(f)

        def generate_ctx_views(x):
            b, v, l, c = x.shape
            ctx_views = x.unsqueeze(1).expand(b, v, v, l, c)
            mask = torch.arange(v).unsqueeze(0) != torch.arange(v).unsqueeze(1)
            ctx_views = ctx_views[:, mask].reshape(b, v, v - 1, l, c)  # B, V, V-1, L, C
            ctx_views = ctx_views.flatten(2, 3)  # B, V, (V-1)*L, C
            return ctx_views.contiguous()

        pos_ctx = generate_ctx_views(pose)
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            feat_current = final_output[-1]
            feat_current_ctx = generate_ctx_views(feat_current)
            # img1 side
            f1, _ = blk1(
                feat_current[:, 0].contiguous(),
                feat_current_ctx[:, 0].contiguous(),
                pose[:, 0].contiguous(),
                pos_ctx[:, 0].contiguous(),
            )
            f1 = f1.unsqueeze(1)
            # img2 side
            f2, _ = blk2(
                rearrange(feat_current[:, 1:], "b v l c -> (b v) l c"),
                rearrange(feat_current_ctx[:, 1:], "b v l c -> (b v) l c"),
                rearrange(pose[:, 1:], "b v l c -> (b v) l c"),
                rearrange(pos_ctx[:, 1:], "b v l c -> (b v) l c"),
            )
            f2 = rearrange(f2, "(b v) l c -> b v l c", b=b, v=v - 1)
            # store the result
            final_output.append(torch.cat((f1, f2), dim=1))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        last_feat = rearrange(final_output[-1], "b v l c -> (b v) l c")
        last_feat = self.dec_norm(last_feat)
        final_output[-1] = rearrange(last_feat, "(b v) l c -> b v l c", b=b, v=v)
        return final_output

    def forward(
        self,
        context: dict,
        symmetrize_batch=False,
        return_views=True,
    ):
        b, v, _, h, w = context["image"].shape
        images_all = context["image"]

        # camera embedding in the encoder
        intrinsic_embedding_all = None
        intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2))
        intrinsic_embedding_all = rearrange(
            intrinsic_embedding, "b v c -> (b v) c"
        ).unsqueeze(1)

        # step 1: encoder input images
        images_all = rearrange(images_all, "b v c h w -> (b v) c h w")
        shape_all = torch.tensor(images_all.shape[-2:])[None].repeat(b * v, 1)

        feat, pose, _, all_feat = self._encode_image(
            images_all, shape_all, intrinsic_embedding_all
        )

        feat = rearrange(feat, "(b v) l c -> b v l c", b=b, v=v)
        pose = rearrange(pose, "(b v) l c -> b v l c", b=b, v=v)

        # step 2: decoder
        if hasattr(self, "dec_blocks"):
            dec_feat = self._decoder(feat, pose)
            shape = rearrange(shape_all, "(b v) c -> b v c", b=b, v=v)
            images = rearrange(images_all, "(b v) c h w -> b v c h w", b=b, v=v)
            dec_feat = list(dec_feat)
            for i in range(len(dec_feat)):
                dec_feat[i] = dec_feat[i][:, :, :-1]
        else:
            dec_feat = None
            shape = rearrange(shape_all, "(b v) c -> b v c", b=b, v=v)
            images = rearrange(images_all, "(b v) c h w -> b v c h w", b=b, v=v)

        feat = feat[:, :, :-1, :]  # b,v,l,c
        for i in range(len(all_feat)):
            all_feat[i] = rearrange(all_feat[i], "(b v) l c -> b v l c", b=b, v=v)
            all_feat[i] = all_feat[i][:, :, :-1, :]

        return feat, all_feat, dec_feat, shape, images

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024


class CroCoEncoderOnly(CroCoNet):
    def __init__(
        self,
        enc_depth=24,
        dec_depth=12,
        enc_embed_dim=1024,
        dec_embed_dim=768,
        enc_num_heads=16,
        dec_num_heads=12,
        pos_embed="RoPE100",
        img_size=(480, 640),  # note that this is ignored, but kept for compatibility
        patch_size=16,
        freeze="none",
    ) -> None:
        super().__init__(
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            enc_embed_dim=enc_embed_dim,
            dec_embed_dim=dec_embed_dim,
            enc_num_heads=enc_num_heads,
            dec_num_heads=dec_num_heads,
            pos_embed=pos_embed,
            img_size=img_size,
            patch_size=patch_size,
        )
        # self.intrinsic_encoder = nn.Linear(9, enc_embed_dim)  # 3x3 intrinsics matrix

        self.set_freeze(freeze)

    def _set_patch_embed(
        self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3
    ):
        # note there img_size will be ignored
        self.patch_embed = PatchEmbedDust3R(
            img_size,
            patch_size,
            in_chans,
            enc_embed_dim,
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        pass

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks2") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks2")] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ["none", "encoder"], f"unexpected freeze={freeze}"
        to_be_frozen = {
            "none": [],
            "encoder": [self.patch_embed, self.enc_blocks],
        }
        log.info(f"Freezing {freeze}")
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        pass

    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = None

    def _set_mask_generator(self, num_patches, mask_ratio):
        """No mask generator"""
        pass

    def _encode_image(self, image, true_shape, intrinsics_embed=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        if intrinsics_embed is not None:
            x = torch.cat((x, intrinsics_embed), dim=1)
            add_pose = pos[:, 0:1, :].clone()
            add_pose[:, :, 0] += pos[:, -1, 0].unsqueeze(-1) + 1
            pos = torch.cat((pos, add_pose), dim=1)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        out = []
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)
            out.append(x)

        x = self.enc_norm(x)
        return x, pos, None, out

    def _encode_image_pairs(
        self,
        img1,
        img2,
        true_shape1,
        true_shape2,
        intrinsics_embed1=None,
        intrinsics_embed2=None,
    ):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _, all_feat = self._encode_image(
                torch.cat((img1, img2), dim=0),
                torch.cat((true_shape1, true_shape2), dim=0),
                (
                    torch.cat((intrinsics_embed1, intrinsics_embed2), dim=0)
                    if intrinsics_embed1 is not None
                    else None
                ),
            )
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
            all_feat1 = []
            all_feat2 = []
            for feat in all_feat:
                feat1, feat2 = feat.chunk(2, dim=0)
                all_feat1.append(feat1)
                all_feat2.append(feat2)
        else:
            out, pos, _, all_feat1 = self._encode_image(
                img1, true_shape1, intrinsics_embed1
            )
            out2, pos2, _, all_feat2 = self._encode_image(
                img2, true_shape2, intrinsics_embed2
            )
        return out, out2, pos, pos2, all_feat1, all_feat2

    def _encode_symmetrized(self, view1, view2, force_asym=False):
        img1 = view1["img"]
        img2 = view2["img"]
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get(
            "true_shape", torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        )
        shape2 = view2.get(
            "true_shape", torch.tensor(img2.shape[-2:])[None].repeat(B, 1)
        )
        # warning! maybe the images have different portrait/landscape orientations
        # intrinsics_embed1 = view1.get("intrinsics_embed", None)
        # intrinsics_embed2 = view2.get("intrinsics_embed", None)

        if force_asym or not is_symmetrized(view1, view2):
            feat1, feat2, pos1, pos2, all_feat1, all_feat2 = self._encode_image_pairs(
                img1, img2, shape1, shape2
            )
        else:
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2, all_feat1, all_feat2 = self._encode_image_pairs(
                img1[::2], img2[::2], shape1[::2], shape2[::2]
            )
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
            all_feat1 = map(interleave, all_feat1)
            all_feat2 = map(interleave, all_feat2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2), (all_feat1, all_feat2)

    def _decoder(self, f1, pos1, f2, pos2, extra_embed1=None, extra_embed2=None):
        pass

    def _downstream_head(self, head_num, decout, img_shape):
        pass

    def forward(
        self,
        context: dict,
        symmetrize_batch=False,
        return_views=False,
    ):
        b, v, _, h, w = context["image"].shape
        device = context["image"].device

        view1, view2 = (
            {"img": context["image"][:, 0]},
            {"img": context["image"][:, 1]},
        )

        # camera embedding in the encoder
        # intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2))
        # view1["intrinsics_embed"] = intrinsic_embedding[:, 0].unsqueeze(1)
        # view2["intrinsics_embed"] = intrinsic_embedding[:, 1].unsqueeze(1)

        if symmetrize_batch:
            instance_list_view1, instance_list_view2 = (
                [0 for _ in range(b)],
                [1 for _ in range(b)],
            )
            view1["instance"] = instance_list_view1
            view2["instance"] = instance_list_view2
            view1["idx"] = instance_list_view1
            view2["idx"] = instance_list_view2
            view1, view2 = make_batch_symmetric(view1, view2)

            # encode the two images --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2), (all_feat1, all_feat2) = (
                self._encode_symmetrized(view1, view2, force_asym=False)
            )
        else:
            # encode the two images --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2), (all_feat1, all_feat2) = (
                self._encode_symmetrized(view1, view2, force_asym=True)
            )

        if return_views:
            return (
                feat1,
                feat2,
                all_feat1,
                all_feat2,
                None,
                None,
                shape1,
                shape2,
                view1,
                view2,
            )
        return (
            feat1,
            feat2,
            all_feat1,
            all_feat2,
            None,
            None,
            shape1,
            shape2,
        )

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024
