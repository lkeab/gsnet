[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gsnet-joint-vehicle-pose-and-shape/vehicle-pose-estimation-on-apollocar3d)](https://paperswithcode.com/sota/vehicle-pose-estimation-on-apollocar3d?p=gsnet-joint-vehicle-pose-and-shape)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gsnet-joint-vehicle-pose-and-shape/3d-shape-reconstruction-on-apollocar3d)](https://paperswithcode.com/sota/3d-shape-reconstruction-on-apollocar3d?p=gsnet-joint-vehicle-pose-and-shape)

# GSNet: Joint Vehicle Pose and Shape Reconstruction with Geometrical and Scene-aware Supervision
Code and 3D car mesh models for the *ECCV 2020 paper* "GSNet: Joint Vehicle Pose and Shape Reconstruction with Geometrical and Scene-aware Supervision".
GSNet performs joint vehicle pose estimation and vehicle shape reconstruction with single RGB image as input.

### [[arXiv](https://arxiv.org/abs/2007.13124)]|[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600511.pdf)]|[[Project Page](http://lkeab.github.io/gsnet/)]

<p align='center'>
<img src='https://github.com/lkeab/gsnet/blob/master/images/framework.png' width='900'/>
</p>

## Abstract
We present a novel end-to-end framework named as **GSNet** (Geometric and Scene-aware Network), which **jointly** estimates 6DoF poses and reconstructs detailed 3D car shapes from single urban street view. GSNet utilizes a unique four-way feature extraction and fusion scheme and directly regresses 6DoF vehicle poses and shapes in a single forward pass. Extensive experiments show that our diverse feature extraction and fusion scheme can greatly improve model performance. Based on a divide-and-conquer 3D shape representation strategy, GSNet reconstructs 3D vehicle shape with great detail (1352 vertices and 2700 faces). This dense mesh representation further leads us to consider geometrical consistency and scene context, and inspires a new multi-objective loss function to regularize network training, which in turn improves the accuracy of 6D pose estimation and validates the merit of jointly performing both tasks. 

Results on ApolloCar3D benchmark
----------
(Check Table 3 of the paper for full results)
| Method  | A3DP-Rel-mean | A3DP-Abs-mean |
|----------|--------|-----------|
| DeepMANTA (CVPR'17) | 16.04 | 20.1 |
| 3D-RCNN (CVPR'18) | 10.79 | 16.44 |
| Kpt-based (CVPR'19) | 16.53 | 20.4 |
| Direct-based (CVPR'19) | 11.49 | 15.15 |
| **GSNet (ECCV'20)** | **20.21** | **18.91**|


## Installation
We build GSNet based on the [Detectron2](https://github.com/facebookresearch/detectron2/) developed by FAIR. Please first follow its [readme file](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). We recommend the Pre-Built Detectron2 (Linux only) version with pytorch 1.5 by the following command:

```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

## Dataset Preparation
The ApolloCar3D dataset is detailed in paper [ApolloCar3D](https://openaccess.thecvf.com/content_CVPR_2019/papers/Song_ApolloCar3D_A_Large_3D_Car_Instance_Understanding_Benchmark_for_Autonomous_CVPR_2019_paper.pdf) and the corresponding images can be obtained from [link](http://apolloscape.auto/car_instance.html).
We provide our converted car meshes (same topology), kpts, bounding box, 3d pose annotations etc. in coco format under the [car_deform_result](https://github.com/lkeab/gsnet/blob/master/car_deform_result/) and [datasets/apollo/annotations](https://github.com/lkeab/gsnet/blob/master/datasets/apollo/annotations/) folders.

## Environment
- Python 3.6
- Numpy 1.16
- PyTorch >= 1.0.1
- CUDA 9/10
- [Softras](https://github.com/ShichenLiu/SoftRas)
- [Pyrender](https://github.com/mmatl/pyrender)

## Using Our Car mesh models
[car_deform_result](https://github.com/lkeab/gsnet/blob/master/car_deform_result/): We provide 79 types of ground truth car meshes with the **same topology** (1352 vertices and 2700 faces) converted using SoftRas (https://github.com/ShichenLiu/SoftRas) 

The file [car_models.py](https://github.com/lkeab/gsnet/blob/master/car_deform_result/car_models.py) has a detailed description on the car id and car type correspondance.

[merge_mean_car_shape](https://github.com/lkeab/gsnet/blob/master/merge_mean_car_shape/): The mean car shape of the four shape basis used by four independent PCA models.

[pca_components](https://github.com/lkeab/gsnet/blob/master/pca_components): The learned weights of the four PCA models.

![Image of GSNet shape reconstruction](https://github.com/lkeab/gsnet/blob/master/images/shape_reconstruction.png)

**How to use our car mesh models?** Please refer to the `class StandardROIHeads` in [roi_heads.py](https://github.com/lkeab/gsnet/blob/master/reference_code/roi_heads.py), which contains the core inference code for ROI head of GSNet. It relies on the [SoftRas](https://github.com/ShichenLiu/SoftRas) to load and manipulate the car meshes.

## Run GSNet
Please follow the [readme](https://github.com/lkeab/gsnet/tree/master/reference_code/GSNet-release) page (including the pretrained model).

## Citation
Please star this repository and cite the following paper in your publications if it helps your research:

    @InProceedings{Ke_2020_ECCV,
        author = {Ke, Lei and Li, Shichao and Sun, Yanan and Tai, Yu-Wing and Tang, Chi-Keung},
        title = {GSNet: Joint Vehicle Pose and Shape Reconstruction with Geometrical and Scene-aware Supervision},
        booktitle = {European Conference on Computer Vision (ECCV)},
        year = {2020}
    }

## License
A MIT license is used for this repository. However, certain third-party datasets, such as (ApolloCar3D), are subject to their respective licenses and may not grant commercial use.
