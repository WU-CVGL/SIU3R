<h2 align="center"> <a href="https://arxiv.org/abs/2507.02705"> SIU3R: Simultaneous Scene Understanding and 3D Reconstruction Beyond Feature Alignment

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2507.02705-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.02705)
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://insomniaaac.github.io/siu3r/)

</h5>
<div align="center">
This repository is the official implementation of the SIU3R.

SIU3R is a feed-forward method that can achieve simultaneous 3D scene understanding and reconstruction given unposed images. In particular, SIU3R does not require feature alignment with 2D VFMs (e.g., CLIP, LSeg) to enable understanding, which unleashes its potential as a unified model to achieve multiple 3D understanding tasks (i.e., semantic, instance, panoptic and text-referred segmentation). Moreover, tailored designs for mutual benefits can further boost SIU3R's performance by encouraging bi-directional promotion between reconstruction and understanding.
</div>
<br>

https://github.com/user-attachments/assets/95034781-75e4-4317-ab34-a9ea4ed7a644

## 🛠️ Installation
We recommend using [uv](https://docs.astral.sh/uv/) to create a virtual environment for this project. The following instructions assume you have `uv` installed. Our code is tested with Python 3.10 and PyTorch 2.4.1 with cuda 11.8.

To set up the environment, just run `uv sync` command.

## ⚡️ Inference
To run inference, you can download the pre-trained model from [here](https://huggingface.co/datasets/insomnia7/SIU3R/blob/main/siu3r_epoch100.ckpt) and place it in the `pretrained_weights` directory.

Then, you can run the inference script:
```bash
python inference.py --image_path1 <path_to_image1> --image_path2 <path_to_image2> --output_path <output_directory> [--cx <cx_value>] [--cy <cy_value>] [--fx <fx_value>] [--fy <fy_value>]
```
A `output.ply` will be generated in the specified output directory, containing the reconstructed gaussian splattings. The `cx`, `cy`, `fx`, and `fy` parameters are optional and can be used to specify the camera intrinsics. If not provided, default values will be used.

You can view the results in the online viewer by running:
```bash
python viewer.py --output_ply <output_directory/output.ply>
```

## 📚 Dataset
We use the ScanNet dataset for training and evaluation. You can download the processed dataset from [here](https://huggingface.co/datasets/insomnia7/SIU3R/tree/main/scannet) and place it in the `data` directory. The dataset should have the following structure:
```
data/
├── scannet/
│   ├── train/
|   |   |-- scene0000_00
|   |   |   |-- color
|   |   |   |-- depth
|   |   |   |-- extrinsic
|   |   |   |-- instance
|   |   |   |-- intrinsic.txt
|   |   |   |-- iou.png
|   |   |   |-- iou.pt
|   |   |   |-- panoptic
|   |   |   `-- semantic
|   |   `-- ....
|   └── val/
│       ├── scene0011_00
│       │   |-- color
│       │   |-- depth
│       │   |-- extrinsic
│       │   |-- instance
│       │   |-- intrinsic.txt
│       │   |-- iou.png
│       │   |-- iou.pt
│       │   |-- panoptic
│       │   `-- semantic
│       `-- ....
|-- train_refer_seg_data.json
|-- val_pair.json
|-- val_refer_pair.json
`-- val_refer_seg_data.json
```

## 📝 Training
If you want to train the model, you should download pretrained MASt3R weights from [here](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth), our pretrained panoptic segmentation head weights from [here](https://huggingface.co/datasets/insomnia7/SIU3R/blob/main/panoptic_coco_pretrain_vitadapter_maskdecoder_epoch60.ckpt) and put them in the `pretrained_weights` directory. 

To train the model, you can use the following command:
```bash
python src/run.py experiment=siu3r_train
```
This will start the training process using the configuration specified in `configs/main.yaml`. You can modify the configuration file to adjust the training parameters, such as devices, learning rate, batch size, and number of epochs.

## 📷 Camera Conventions
Our camera system is the same as [pixelSplat](https://github.com/dcharatan/pixelsplat). The camera intrinsic matrices are normalized (the first row is divided by image width, and the second row is divided by image height). The camera extrinsic matrices are OpenCV-style camera-to-world matrices ( +X right, +Y down, +Z camera looks into the screen).

## 📖 Citation
If you find our work useful, please consider citing our paper:
```bibtex
@misc{xu2025siu3r,
      title={SIU3R: Simultaneous Scene Understanding and 3D Reconstruction Beyond Feature Alignment}, 
      author={Qi Xu and Dongxu Wei and Lingzhe Zhao and Wenpu Li and Zhangchi Huang and Shunping Ji and Peidong Liu},
      year={2025},
      eprint={2507.02705},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.02705}, 
}
```
