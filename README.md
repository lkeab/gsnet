# GSNet: Joint Vehicle Pose and Shape Reconstruction with Geometrical and Scene-aware Supervision
Relevant code and 3D car mesh models for the ECCV 2020 paper "GSNet: Joint Vehicle Pose and Shape Reconstruction with Geometrical and Scene-aware Supervision" (coming soon).

arXiv: https://arxiv.org/abs/2007.13124

![Image of GSNet framework](https://github.com/lkeab/gsnet/blob/master/images/framework.png)

## Car mesh models
[car_deform_result](https://github.com/lkeab/gsnet/blob/master/car_deform_result/): The 79 types of ground truth car meshes with the same topology (1352 vertices and 2700 faces) converted using SoftRas (https://github.com/ShichenLiu/SoftRas) 

The file [car_models.py](https://github.com/lkeab/gsnet/blob/master/car_deform_result/car_models.py) has a more detailed description.

merge_mean_car_shape: The four shape basis obtained by four independent PCA models.

![Image of GSNet shape reconstruction](https://github.com/lkeab/gsnet/blob/master/images/shape_reconstruction.png)
