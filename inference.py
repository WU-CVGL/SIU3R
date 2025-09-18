from PIL import Image
import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.pipeline import Pipeline
from src.utils.ply_export import export_ply


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    # resize shortest side to 256 and then center crop to 256x256
    if W < H:
        new_W = 256
        new_H = int(H * (256 / W))
        image = image.resize((new_W, new_H), Image.Resampling.LANCZOS)
        left = 0
        top = (new_H - 256) // 2
        right = new_W
        bottom = top + 256
        image = image.crop((left, top, right, bottom))
    else:
        new_H = 256
        new_W = int(W * (256 / H))
        image = image.resize((new_W, new_H), Image.Resampling.LANCZOS)
        left = (new_W - 256) // 2
        top = 0
        right = left + 256
        bottom = new_H
        image = image.crop((left, top, right, bottom))
    # convert to numpy array and normalize to [0, 1]
    image = np.array(image).astype(np.float32)
    image = torch.from_numpy(image).permute(2, 0, 1) / 255.0
    return image


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="pretrained_weights/siu3r_epoch100.ckpt",
        help="Path to the model file.",
    )
    parser.add_argument(
        "--image_path1",
        type=str,
        default="assets/living_room_image1.jpg",
        help="Path to the first image file.",
    )
    parser.add_argument(
        "--image_path2",
        type=str,
        default="assets/living_room_image2.jpg",
        help="Path to the second image file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="infer_outputs",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=128.0,
        help="Camera intrinsic cx",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=128.0,
        help="Camera intrinsic cy",
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=318.0,
        help="Camera intrinsic fx",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=318.0,
        help="Camera intrinsic fy",
    )
    args = parser.parse_args()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    image_path1 = Path(args.image_path1)
    image_path2 = Path(args.image_path2)
    if not image_path1.exists():
        raise FileNotFoundError(f"Image file {image_path1} does not exist.")
    if not image_path2.exists():
        raise FileNotFoundError(f"Image file {image_path2} does not exist.")
    cx, cy, fx, fy = args.cx, args.cy, args.fx, args.fy
    image1 = preprocess_image(image_path1)
    image2 = preprocess_image(image_path2)
    images = torch.stack([image1, image2], dim=0).unsqueeze(0)  # [1, 2, 3, H, W]
    intrinsics = torch.tensor(
        [
            [
                [fx / 256.0, 0, cx / 256.0],
                [0, fy / 256.0, cy / 256.0],
                [0, 0, 1],
            ]
        ]
    ).repeat(1, 2, 1, 1)  # [1, 2, 3, 3]
    if torch.cuda.is_available():
        images = images.cuda()
        intrinsics = intrinsics.cuda()
    pipeline = Pipeline.load_from_checkpoint(
        model_path, map_location="cpu", strict=False
    )
    pipeline.eval()
    if torch.cuda.is_available():
        pipeline.cuda()
    with torch.no_grad():
        (
            gaussians,
            context_seg_output,
            context_seg_masks,
            context_seg_infos,
            context_seg_query_scores,
        ) = pipeline.model(
            images,
            intrinsics,
            enable_query_class_logit_lift=True,
        )
    gaussians = gaussians.detach_cpu_copy()
    export_ply(
        means=gaussians.means[0],
        scales=gaussians.scales[0],
        rotations=gaussians.rotations[0],
        harmonics=gaussians.harmonics[0],
        opacities=gaussians.opacities[0],
        semantic_labels=gaussians.semantic_labels[0],
        instance_labels=gaussians.instance_labels[0],
        seg_query_class_logits=gaussians.seg_query_class_logits[0],
        path=output_path / "output.ply",
        shift_and_scale=False,
        save_sh_dc_only=False,
    )
