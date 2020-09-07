# GSNet: Joint Vehicle Pose and Shape Reconstruction with Geometrical and Scene-aware Supervision
Relevant code and 3D car mesh models for the *ECCV 2020 paper* "GSNet: Joint Vehicle Pose and Shape Reconstruction with Geometrical and Scene-aware Supervision" (coming soon).

arXiv: https://arxiv.org/abs/2007.13124

Paper: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600511.pdf

project page: http://lkeab.github.io/gsnet/
![Image of GSNet framework](https://github.com/lkeab/gsnet/blob/master/images/framework.png)

## Abstract
We present a novel end-to-end framework named as **GSNet** (Geometric and Scene-aware Network), which **jointly** estimates 6DoF poses and reconstructs detailed 3D car shapes from single urban street view. GSNet utilizes a unique four-way feature extraction and fusion scheme and directly regresses 6DoF poses and shapes in a single forward pass. Extensive experiments show that our diverse feature extraction and fusion scheme can greatly improve model performance. Based on a divide-and-conquer 3D shape representation strategy, GSNet reconstructs 3D vehicle shape with great detail (1352 vertices and 2700 faces). This dense mesh representation further leads us to consider geometrical consistency and scene context, and inspires a new multi-objective loss function to regularize network training, which in turn improves the accuracy of 6D pose estimation and validates the merit of jointly performing both tasks. 

## Using Car mesh models
[car_deform_result](https://github.com/lkeab/gsnet/blob/master/car_deform_result/): The 79 types of ground truth car meshes with the **same topology** (1352 vertices and 2700 faces) converted using SoftRas (https://github.com/ShichenLiu/SoftRas) 

The file [car_models.py](https://github.com/lkeab/gsnet/blob/master/car_deform_result/car_models.py) has a detailed description on the car id and car type correspondance.

[merge_mean_car_shape](https://github.com/lkeab/gsnet/blob/master/merge_mean_car_shape/): The mean car shape of the four shape basis used by four independent PCA models.

[pca_components](https://github.com/lkeab/gsnet/blob/master/pca_components): The learned weights of the four PCA models.

![Image of GSNet shape reconstruction](https://github.com/lkeab/gsnet/blob/master/images/shape_reconstruction.png)

**How to use the our car mesh models?** Please refer to the [roi_heads.py](https://github.com/lkeab/gsnet/blob/master/reference_code/roi_heads.py), which contains the core inference code for ROI head of GSNet. It relies on the [SoftRas](https://github.com/ShichenLiu/SoftRas) to load and manipulate the car meshes.

## Installation
We build GSNet based on the [Detectron2](https://github.com/facebookresearch/detectron2/) developed by FAIR. Please first follow its [readme file](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). We recommend the Pre-Built Detectron2 (Linux only) version with pytorch 1.5. I use the Pre-Built Detectron2 with CUDA 10.1 and pytorch 1.5 and you can run this code to install it.

```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

## Environment
- Python 3.6
- Numpy 1.16
- PyTorch >= 1.0.1
- CUDA 9/10
- Softras
- Pyrender

## Citation
Please star this repository and cite the following paper in your publications if it helps your research:

    @InProceedings{Ke_2020_ECCV,
    author = {Ke, Lei and Li, Shichao and Sun, Yanan and Tai, Yu-Wing and Tang, Chi-Keung},
    title = {GSNet: Joint Vehicle Pose and Shape Reconstruction with Geometrical and Scene-aware Supervision},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    year = {2020}
    }
