from typing import Literal
from math import isqrt
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from src.utils.gaussians_types import Gaussians
from gsplat import rasterization
from src.models.cuda_splatting import render_cuda


class SplattingCUDA(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.near = 0.1
        self.far = 100.0
        self.scale_factor = 1 / self.near
        self.register_buffer(
            "background_color",
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        image_shape: tuple[int, int],
        render_color: bool = True,
        render_feature: bool = False,
        render_id: bool = False,
        render_qc_logits: bool = False,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ):
        b, v, _, _ = extrinsics.shape
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * self.scale_factor
        gaussians.covariances *= self.scale_factor**2
        gaussians.means *= self.scale_factor
        near = 1.0
        far = self.far * self.scale_factor
        if render_color:
            color, depth = render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                torch.tensor(near, dtype=torch.float32, device="cuda").repeat(b * v),
                torch.tensor(far, dtype=torch.float32, device="cuda").repeat(b * v),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                cam_rot_delta=(
                    rearrange(cam_rot_delta, "b v i -> (b v) i")
                    if cam_rot_delta is not None
                    else None
                ),
                cam_trans_delta=(
                    rearrange(cam_trans_delta, "b v i -> (b v) i")
                    if cam_trans_delta is not None
                    else None
                ),
            )
            color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
            color = torch.clamp(color, 0.0, 1.0)
            depth = rearrange(depth, "(b v) h w -> b v h w", b=b, v=v)
        if render_qc_logits:
            width = image_shape[1]
            height = image_shape[0]
            seg_query_class_logits = gaussians.seg_query_class_logits  # b * [n, q, c]
            all_query_class_logits = []
            # iterate over the batch
            for i in range(b):
                means = gaussians.means[i]
                covariances = gaussians.covariances[i]
                opacities = gaussians.opacities[i]
                Ks = intrinsics[i].clone()
                Ks[:, 0, :] *= width
                Ks[:, 1, :] *= height
                viewmats = torch.linalg.inv(extrinsics[i])
                query_class_logits = seg_query_class_logits[i]
                _, q, c = query_class_logits.shape
                query_class_logits = rearrange(query_class_logits, "n q c -> n (q c)")
                rendered_qc_logits, _, _ = rasterization(
                    means=means,
                    quats=None,
                    scales=None,
                    covars=covariances,
                    opacities=opacities,
                    colors=query_class_logits,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=None,
                    near_plane=near,
                    far_plane=far,
                )
                rendered_qc_logits = rearrange(
                    rendered_qc_logits, "n h w (q c) -> n q c h w", q=q, c=c
                )
                all_query_class_logits.append(rendered_qc_logits)

        return {
            "render_color": color if render_color else None,
            "render_depth": depth if render_color else None,
            "render_qc_logits": all_query_class_logits if render_qc_logits else None,
        }
