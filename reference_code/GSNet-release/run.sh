python3 setup.py build develop #--no-deps

CUDA_VISIBLE_DEVICES=0 python3 demo/demo.py --config-file configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x_apollo.yaml \
  --input 'datasets/coco/val_apollo/' \
  --output 'medium_result_val/' \
  --opts MODEL.WEIGHTS ./output/model_final.pth

#detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
