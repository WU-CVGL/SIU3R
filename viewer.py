import argparse
from pathlib import Path
from typing import Dict, Literal, Tuple
import time
import numpy as np
import torch
from torch import Tensor
import viser
import viser.transforms as vtf
import nerfview
from nerfview.viewer import Viewer
from plyfile import PlyData
from gsplat.rendering import rasterization


class PoseViewer(Viewer):
    def init_scene(
        self,
        pil_images,
        c2ws,
        fov_deg=50,
    ):
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.original_c2w: Dict[int, np.ndarray] = {}

        total_num = len(pil_images)
        # NOTE: not constraining the maximum number of camera frustums shown
        image_indices = np.linspace(
            0, total_num - 1, total_num, dtype=np.int32
        ).tolist()
        for idx in image_indices:
            image_uint8 = np.asarray(pil_images[idx].resize((256, 256)))
            R = vtf.SO3.from_matrix(c2ws[idx][:3, :3])
            # NOTE: not understand why this is needed in nerfstudio viewer, but comment it out make ours work
            # probably because gsplat uses OpenCV convention, whereas nerfstudio use the Blender / OpenGL convention
            # R = R @ vtf.SO3.from_x_radians(np.pi)

            camera_handle = self.server.scene.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=np.deg2rad(fov_deg),
                scale=0.5,  # hardcode this scale for now
                aspect=1,
                image=image_uint8,
                wxyz=R.wxyz,
                position=c2ws[idx][:3, 3],
                # NOTE: not multiplied by VISER_NERFSTUDIO_SCALE_RATIO, this should also be used in get_camera_state
            )

            @camera_handle.on_click
            def _(
                event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle],
            ) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles[idx] = camera_handle
            self.original_c2w[idx] = c2ws[idx]

        self.state.status = "test"


class GaussianRenderer:
    def __init__(
        self,
        ply_path,
        num_classes=21,
        H=256,
        W=256,
        semantic_threshold=0.1,
        port=12025,
    ):
        self.num_classes = num_classes
        self.H = H
        self.W = W
        self.semantic_threshold = semantic_threshold
        self.load_ply(ply_path)

        # viewer
        self.server = viser.ViserServer(port=port, verbose=False)
        self.semantic_button = self.server.gui.add_button(
            label="show semantic",
        )
        self.semantic_button.on_click(lambda _: self.semantic_button_callback())
        self.instance_button = self.server.gui.add_button(
            label="show instance",
        )
        self.instance_button.on_click(lambda _: self.instance_button_callback())

        self.rgb_button = self.server.gui.add_button(
            label="show rgb",
        )
        self.rgb_button.on_click(lambda _: self.rgb_button_callback())

        self.viewer = PoseViewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
        )

        self.color_map = {
            0: np.array([255, 255, 255], dtype=np.uint8),  # unlabeled
            1: np.array([174, 199, 232], dtype=np.uint8),  # wall
            2: np.array([152, 223, 138], dtype=np.uint8),  # floor
            3: np.array([31, 119, 180], dtype=np.uint8),  # cabinet
            4: np.array([255, 187, 120], dtype=np.uint8),  # bed
            5: np.array([188, 189, 34], dtype=np.uint8),  # chair
            6: np.array([140, 86, 75], dtype=np.uint8),  # sofa
            7: np.array([255, 152, 150], dtype=np.uint8),  # table
            8: np.array([214, 39, 40], dtype=np.uint8),  # door
            9: np.array([197, 176, 213], dtype=np.uint8),  # window
            10: np.array([148, 103, 189], dtype=np.uint8),  # bookshelf
            11: np.array([196, 156, 148], dtype=np.uint8),  # picture
            12: np.array([23, 190, 207], dtype=np.uint8),  # counter
            13: np.array([247, 182, 210], dtype=np.uint8),  # desk
            14: np.array([219, 219, 141], dtype=np.uint8),  # curtain
            15: np.array([255, 127, 14], dtype=np.uint8),  # refrigerator
            16: np.array([158, 218, 229], dtype=np.uint8),  # shower curtain
            17: np.array([44, 160, 44], dtype=np.uint8),  # toilet
            18: np.array([112, 128, 144], dtype=np.uint8),  # sink
            19: np.array([227, 119, 194], dtype=np.uint8),  # bathtub
            20: np.array([82, 84, 163], dtype=np.uint8),  # otherfurn
        }
        self.ins_color_map = {i: np.random.rand(3) for i in range(200)}

    def semantic_button_callback(self):
        self.viewer.render_fn = self._semantic_render_fn

    def instance_button_callback(self):
        self.viewer.render_fn = self._instance_render_fn

    def rgb_button_callback(self):
        self.viewer.render_fn = self._viewer_render_fn

    def load_ply(
        self,
        path,
        crop=True,
    ):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

        if len(extra_f_names) == 0:
            self.max_sh_degree = 0
        if len(extra_f_names) == 9:
            self.max_sh_degree = 1
        if len(extra_f_names) == 24:
            self.max_sh_degree = 2
        if len(extra_f_names) == 45:
            self.max_sh_degree = 3
        if len(extra_f_names) == 72:
            self.max_sh_degree = 4
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        semantic_label = np.asarray(plydata.elements[0]["semantic_label"])
        instance_label = np.asarray(plydata.elements[0]["instance_label"])

        qc_logit_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("seg_query_class_logits_")
        ]
        qc_logits = np.zeros((xyz.shape[0], len(qc_logit_names)))
        num_queries = len(qc_logit_names) // self.num_classes
        for idx, attr_name in enumerate(qc_logit_names):
            qc_logits[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        rots = torch.tensor(rots, dtype=torch.float, device="cuda")
        scales = torch.tensor(scales, dtype=torch.float, device="cuda")
        opacities = torch.tensor(opacities, dtype=torch.float, device="cuda")
        features_dc = (
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
        )
        features_extra = (
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
        )
        semantic_label = torch.tensor(semantic_label, dtype=torch.long, device="cuda")
        instance_label = torch.tensor(instance_label, dtype=torch.long, device="cuda")
        qc_logits = torch.tensor(qc_logits, dtype=torch.float, device="cuda").view(
            -1, num_queries, self.num_classes
        )
        H, W = self.H, self.W
        # default crop because not accurate intrinsics lead bad boarder gaussians
        if crop:
            crop_size = 5
            xyz = xyz.view(-1, H, W, 3)
            xyz = xyz[:, crop_size : H - crop_size, crop_size : W - crop_size, :]
            xyz = xyz.reshape(-1, 3)
            rots = rots.view(-1, H, W, 4)
            rots = rots[:, crop_size : H - crop_size, crop_size : W - crop_size, :]
            rots = rots.reshape(-1, 4)
            scales = scales.view(-1, H, W, 3)
            scales = scales[:, crop_size : H - crop_size, crop_size : W - crop_size, :]
            scales = scales.reshape(-1, 3)
            opacities = opacities.view(-1, H, W)
            opacities = opacities[
                :, crop_size : H - crop_size, crop_size : W - crop_size
            ]
            opacities = opacities.reshape(-1)
            features_dc = features_dc.view(-1, H, W, 1, 3)
            features_dc = features_dc[
                :, crop_size : H - crop_size, crop_size : W - crop_size, :, :
            ]
            features_dc = features_dc.reshape(-1, 1, 3)
            features_extra = features_extra.view(
                -1, H, W, (self.max_sh_degree + 1) ** 2 - 1, 3
            )
            features_extra = features_extra[
                :, crop_size : H - crop_size, crop_size : W - crop_size, :, :
            ]
            features_extra = features_extra.reshape(
                -1, (self.max_sh_degree + 1) ** 2 - 1, 3
            )
            semantic_label = semantic_label.view(-1, H, W)
            semantic_label = semantic_label[
                :, crop_size : H - crop_size, crop_size : W - crop_size
            ]
            semantic_label = semantic_label.reshape(-1)
            instance_label = instance_label.view(-1, H, W)
            instance_label = instance_label[
                :, crop_size : H - crop_size, crop_size : W - crop_size
            ]
            instance_label = instance_label.reshape(-1)
            qc_logits = qc_logits.view(-1, H, W, num_queries, self.num_classes)
            qc_logits = qc_logits[
                :, crop_size : H - crop_size, crop_size : W - crop_size, :, :
            ]
            qc_logits = qc_logits.reshape(-1, num_queries, self.num_classes)
            H, W = H - 2 * crop_size, W - 2 * crop_size

        self.splats = dict(
            means=xyz,
            quats=rots,
            scales=scales,
            opacities=opacities,
            sh0=features_dc,
            shN=features_extra,
            semantic_label=semantic_label,
            instance_label=instance_label,
            qc_logits=qc_logits,
        )

        self.active_sh_degree = self.max_sh_degree
        print(
            f"Loaded {path} with {self.splats['means'].shape[0]} splats and max SH degree {self.max_sh_degree}"
        )

    def set_cameras(self, pil_images, c2ws, fov_deg=50):
        self.viewer.init_scene(pil_images, c2ws, fov_deg)

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=True,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            backgrounds=torch.ones(3, dtype=torch.float32).to(camtoworlds.device),
            **kwargs,
        )
        return render_colors, render_alphas, info

    def rasterize_qc_logits(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ):
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        qc_logits = self.splats["qc_logits"]  # [N, num_queries, 21]
        N, num_queries, num_classes = qc_logits.shape
        qc_logits = qc_logits.flatten(start_dim=1)
        rasterize_mode = "classic"
        render_qc_logits, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=qc_logits,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )
        render_qc_logits = render_qc_logits.view(
            -1, height, width, num_queries, num_classes
        ).permute(0, 3, 4, 1, 2)
        return render_qc_logits

    @torch.no_grad()
    def _viewer_render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ):
        # self.viewer._renderers[0]._state = "high"
        self.viewer._renderers[0]._task.action == "static"
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K([width, height])
        c2w = torch.from_numpy(c2w).float().cuda()
        K = torch.from_numpy(K).float().cuda()
        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=self.active_sh_degree,  # active all SH degrees
            radius_clip=0.1,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()

    @torch.no_grad()
    def _qc_logits_render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ):
        # self.viewer._renderers[0]._state = "high"
        self.viewer._renderers[0]._task.action == "static"
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K([width, height])
        c2w = torch.from_numpy(c2w).float().cuda()
        K = torch.from_numpy(K).float().cuda()

        render_qc_logit = self.rasterize_qc_logits(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=None,
            # radius_clip=0.1,  # skip GSs that have small image radius (in pixels)
        )  # [num_queries, num_classes, H, W]
        c_logit, q_index = render_qc_logit.max(dim=1)
        c_logit = torch.concat([c_logit[:, -1:, :, :], c_logit[:, :-1, :, :]], dim=1)
        q_index = torch.concat([q_index[:, -1:, :, :], q_index[:, :-1, :, :]], dim=1)
        sem_logits, sem_id = c_logit.max(dim=1)  # [v, h, w]
        vidx, hidx, widx = torch.meshgrid(
            torch.arange(c_logit.shape[0]),
            torch.arange(height),
            torch.arange(width),
            indexing="ij",
        )
        q_index = q_index[vidx, sem_id, hidx, widx] + 1  # [v, h, w]
        sem_masks = sem_logits < self.semantic_threshold
        sem_id[sem_masks] = 0
        q_index[sem_id == 0] = 0
        for i in [1, 2]:
            q_index[sem_id == i] = 100 + i + 1
        return sem_id.cpu().numpy(), q_index.cpu().numpy()

    def _semantic_render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ):
        sem_id, q_index = self._qc_logits_render_fn(camera_state, render_tab_state)
        color_sem_id = np.ones(
            (sem_id.shape[0], sem_id.shape[1], sem_id.shape[2], 3), dtype=np.float32
        )
        for id in range(self.num_classes):
            # give each semantic class a unique color
            if id not in self.color_map:
                continue
            else:
                color = self.color_map[id] / 255.0
            color_sem_id[sem_id == id] = color
        return color_sem_id

    def _instance_render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ):
        sem_id, q_index = self._qc_logits_render_fn(camera_state, render_tab_state)
        unique_ids = np.unique(q_index)
        colored_instances = np.zeros(
            (q_index.shape[0], q_index.shape[1], q_index.shape[2], 3), dtype=np.float32
        )
        for id in unique_ids:
            # give each instance a unique random color
            if id == 0:
                color = np.array([1, 1, 1], dtype=np.float32)
            else:
                semantic_label = np.unique(sem_id[q_index == id])
                assert len(semantic_label) == 1, (
                    f"multiple semantic_label {semantic_label} for instance {id}"
                )
                semantic_label = semantic_label[0]
                if semantic_label == 1 or semantic_label == 2:
                    color = np.array([1, 1, 1], dtype=np.float32)
                    # color = self.color_map[semantic_label] / 255.0
                else:
                    color = self.color_map[semantic_label] / 255.0
                    color += self.ins_color_map[id] * 0.3
                    color = np.clip(color, 0, 1)
            colored_instances[q_index == id] = color
        return colored_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_ply",
        type=str,
        default="infer_outputs/output.ply",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="Height of the input image.",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="Width of the input image.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=21,
        help="Number of semantic classes.",
    )
    args = parser.parse_args()
    output_ply = Path(args.output_ply)
    if not output_ply.exists():
        raise FileNotFoundError(f"{output_ply} does not exist.")
    print(f"Loading {output_ply}...")

    master = GaussianRenderer(
        output_ply,
        num_classes=args.num_classes,
        H=args.H,
        W=args.W,
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)
