from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    attributes.append("semantic_label")
    attributes.append("instance_label")
    return attributes


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    semantic_labels: Float[Tensor, "gaussian"],
    instance_labels: Float[Tensor, "gaussian"],
    seg_query_class_logits: Float[Tensor, "gaussian num_queries num_classes"],
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
):
    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    # Apply the rotation to the Gaussian rotations.
    # rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    # rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations.detach().cpu().numpy(), "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    list_of_attributes = construct_list_of_attributes(
        0 if save_sh_dc_only else f_rest.shape[1]
    )
    dtype_full = [(attribute, "f4") for attribute in list_of_attributes[:-2]]
    if semantic_labels is not None and instance_labels is not None:
        dtype_full.append(("semantic_label", "i4"))
        dtype_full.append(("instance_label", "i4"))
    if seg_query_class_logits is not None:
        g, q, c = seg_query_class_logits.shape
        seg_query_class_logits = seg_query_class_logits.view(
            g, q * c
        )  # (gaussian, num_queries * num_classes)
        for qc in range(q * c):
            dtype_full.append((f"seg_query_class_logits_{qc}", "f4"))
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    ]
    if semantic_labels is not None and instance_labels is not None:
        attributes.append(semantic_labels[..., None].detach().cpu().numpy())
        attributes.append(instance_labels[..., None].detach().cpu().numpy())
    if seg_query_class_logits is not None:
        attributes.append(seg_query_class_logits.detach().cpu().numpy())
    if save_sh_dc_only:
        # remove f_rest from attributes
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
