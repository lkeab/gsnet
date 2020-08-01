# GSNet: Joint Vehicle Pose and Shape Reconstruction with Geometrical and Scene-aware Supervision
Relevant code and 3D car mesh models for the *ECCV 2020 paper* "GSNet: Joint Vehicle Pose and Shape Reconstruction with Geometrical and Scene-aware Supervision" (coming soon).

arXiv: https://arxiv.org/abs/2007.13124

project page: http://www.kelei.site/gsnet/
![Image of GSNet framework](https://github.com/lkeab/gsnet/blob/master/images/framework.png)

## Abstract
We present a novel end-to-end framework named as **GSNet** (Geometric and Scene-aware Network), which **jointly** estimates 6DoF poses and reconstructs detailed 3D car shapes from single urban street view. GSNet utilizes a unique four-way feature extraction and fusion scheme and directly regresses 6DoF poses and shapes in a single forward pass. Extensive experiments show that our diverse feature extraction and fusion scheme can greatly improve model performance. Based on a divide-and-conquer 3D shape representation strategy, GSNet reconstructs 3D vehicle shape with great detail (1352 vertices and 2700 faces). This dense mesh representation further leads us to consider geometrical consistency and scene context, and inspires a new multi-objective loss function to regularize network training, which in turn improves the accuracy of 6D pose estimation and validates the merit of jointly performing both tasks. 

## Car mesh models
[car_deform_result](https://github.com/lkeab/gsnet/blob/master/car_deform_result/): The 79 types of ground truth car meshes with the **same topology** (1352 vertices and 2700 faces) converted using SoftRas (https://github.com/ShichenLiu/SoftRas) 

The file [car_models.py](https://github.com/lkeab/gsnet/blob/master/car_deform_result/car_models.py) has a detailed description on the car id and car type correspondance.

[merge_mean_car_shape](https://github.com/lkeab/gsnet/blob/master/merge_mean_car_shape/): The four shape basis used by the four independent PCA models.

![Image of GSNet shape reconstruction](https://github.com/lkeab/gsnet/blob/master/images/shape_reconstruction.png)

## Installation
We build GSNet based on the [Detectron2](https://github.com/facebookresearch/detectron2/) developed by FAIR. Please first follow its [readme file](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).
