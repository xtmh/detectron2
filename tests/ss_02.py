from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
import requests

# Load an image
image = cv2.imread("d:\\temp\\human.jpg")
#cv2.imshow('img',image)
#cv2.waitKey(0)

config_file = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75 # Threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
cfg.MODEL.DEVICE = "cuda" # cpu or cuda

# Create predictor
predictor = DefaultPredictor(cfg)

# Make prediction
output = predictor(image)
#print(output)
v = Visualizer(image[:, :, ::-1],
               scale=0.8,
               metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
               instance_mode=ColorMode.IMAGE
               )
v = v.draw_instance_predictions(output["instances"].to("cpu"))
cv2.imshow('images', v.get_image()[:, :, ::-1])
cv2.waitKey(0)