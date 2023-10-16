from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
import pandas as pd
import requests

#   KMP_DUPLICATE_LIB_OK=TRUE 環境変数を設定

# Load an image
#image = cv2.imread("d:\\temp\\office.jpg")
image = cv2.imread("d:\\temp\\tomato2.jpg")
cv2.imshow('img',image)
cv2.waitKey(0)

#学習モデル
#Faster R-CNN
#config_file = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
#Instance Segmentation
#config_file = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
#Keypoint Detection
#config_file = 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
#Panoptic Segmentation
config_file = 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # Threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
cfg.MODEL.DEVICE = "cuda" # cpu or cuda
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True       # For better visualization purpose. Set to False for all classes.

# Create predictor
predictor = DefaultPredictor(cfg)

# Make prediction
panoptic_seg, segments_info = predictor(image)["panoptic_seg"]

metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
v = Visualizer(image[:, :, ::-1],
               scale=1.0,
               metadata=metadata,
               #instance_mode=ColorMode.IMAGE_BW #白黒画像に変換
               instance_mode=ColorMode.IMAGE
               )
#v = v.draw_instance_predictions(output["instances"].to("cpu"))
v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

# imgae show
cv2.imshow('ss_03', v.get_image()[:, :, ::-1])
cv2.waitKey(0)