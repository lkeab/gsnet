# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import os

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image
from detectron2.layers.nms import batched_nms

import soft_renderer as sr
import json

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, save_name):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]

        mask = predictions['instances'].raw_masks.squeeze(1).data.cpu().numpy() if predictions['instances'].has("raw_masks") else None

        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                pred_classes = torch.ones(instances.pred_classes.shape)
                # uncomment to open nms between different classes
                '''
                res = batched_nms(instances.pred_boxes.tensor, instances.scores, pred_classes, 0.5)
                print('res:', res)
                print('res:', res.size()[0])

                #instances.num_instances = res.size()[0]
                instances.pred_boxes.tensor = instances.pred_boxes.tensor[res]
                instances.pred_classes = instances.pred_classes[res]
                instances.scores = instances.scores[res]
                instances.pred_keypoints = instances.pred_keypoints[res]

                instances.predict_trans = instances.predict_trans[res]
                instances.predict_rotation = instances.predict_rotation[res]
                instances.predict_vertices = instances.predict_vertices[res]
                print('pred trans shape:', instances.predict_trans.shape)
                '''

                vis_output = visualizer.draw_instance_predictions(predictions=instances)

                output_trans_dir = './inference_val_translation/'
                output_rotation_dir = './inference_val_rotation/'
                output_mesh_dir = './inference_val_mesh/'
                output_cls_dir = './inference_val_cls/'
                output_score_dir = './inference_val_score/'

                save_name = save_name.split('/')[1]
                template_path = './merge_mean_car_shape/'
                faces = sr.Mesh.from_obj(template_path+'merge_mean_car_model_0.obj').faces

                for directory in [output_trans_dir, output_rotation_dir, output_mesh_dir, output_cls_dir, output_score_dir]:
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                for index in range(instances.predict_trans.shape[0]):
                    with open(os.path.join(output_trans_dir, save_name +'_' +str(index)+'.json'),'w') as f:
                        data = {}
                        data['translation'] = list(instances.predict_trans[index].cpu().detach().numpy().astype(float))
                        json.dump(data, f)

                       
                for index in range(instances.predict_rotation.shape[0]):
                     with open(os.path.join(output_rotation_dir, save_name +'_' +str(index)+'.json'),'w') as f:
                        data = {}
                        data['rotation'] = list(instances.predict_rotation[index].cpu().detach().numpy().astype(float))
                        json.dump(data, f)
                        
                for index in range(instances.pred_classes.shape[0]):
                     with open(os.path.join(output_cls_dir, save_name +'_' +str(index)+'.json'),'w') as f:
                        data = {}
                        data['car_id'] = int(instances.pred_classes[index].cpu().detach().numpy().astype(float))
                        json.dump(data, f)

                for index in range(instances.scores.shape[0]):
                     with open(os.path.join(output_score_dir, save_name +'_' +str(index)+'.json'),'w') as f:
                        data = {}
                        data['score'] = float(instances.scores[index].cpu().detach().numpy().astype(float))
                        json.dump(data, f)
             
                for index in range(instances.predict_vertices.shape[0]):
                    vertices = instances.predict_vertices[index].unsqueeze(0)
                    sr.Mesh(vertices, faces).save_obj(os.path.join(output_mesh_dir, save_name+'_' + str(index) + '.obj'), save_texture=False)
 
        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
