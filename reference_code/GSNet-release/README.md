## Run GSNet
```
bash run.sh
```

## Download the pretrained model 
[Google drive link](https://drive.google.com/file/d/1H2QtfmEl5XeqwYH-kz8phs77BsQGZaku/view?usp=sharing) and put the downloaded beta-version model in the output directory.

## For quantitative evaluation
please use the code provided in https://github.com/ApolloScapeAuto/dataset-api/tree/master/car_instance, and modify the Line628 of the [roi_heads.py](https://github.com/lkeab/gsnet/blob/master/reference_code/roi_heads.py) to:

```
self.test_score_thresh = 0.2
self.test_nms_thresh = 0.75
```
