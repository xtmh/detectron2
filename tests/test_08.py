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
#res = requests.get("https://thumbor.forbes.com/thumbor/fit-in/1200x0/filters%3Aformat%28jpg%29/https%3A%2F%2Fspecials-images.forbesimg.com%2Fimageserve%2F5f15af31465263000625ce08%2F0x0.jpg")
#image = np.asarray(bytearray(res.content), dtype="uint8")
#image = cv2.imdecode(image, cv2.IMREAD_COLOR)

image = cv2.imread("d:\\temp\\office.jpg")
#image = cv2.imread("d:\\temp\\human.jpg")
#image = cv2.imread("d:\\temp\\car03.jpg")
#image = cv2.imread("d:\\temp\\car.png")
#cv2.imshow('img',image)
#cv2.waitKey(0)

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
output = predictor(image)
#panoptic_seg, segments_info = predictor(image)["panoptic_seg"]

# テスト結果を表示
#print(output)
#print(output["instances"].pred_classes)
#print(output["instances"].pred_boxes)

metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
v = Visualizer(image[:, :, ::-1],
               scale=1.0,
               metadata=metadata,
               #instance_mode=ColorMode.IMAGE_BW #白黒画像に変換
               instance_mode=ColorMode.IMAGE
               )
v = v.draw_instance_predictions(output["instances"].to("cpu"))
#v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
#cv2.imshow('test',v.get_image()[:,:,::-1])
#cv2.waitKey(0)

##############################################################################
#result = []
#metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
#[result.extend((#x,
#       panoptic_seg["panoptic_seg"][x].pred_classes.item(),      #class_id
#       [metadata.thing_classes[x] for x in output["panoptic_seg"][x].pred_classes][0], #class
#       panoptic_seg["panoptic_seg"][x].scores.item(),     
#       panoptic_seg["panoptic_seg"][x].pred_boxes.tensor.cpu().numpy()[0][0],
#       panoptic_seg["panoptic_seg"][x].pred_boxes.tensor.cpu().numpy()[0][1],
#       panoptic_seg["panoptic_seg"][x].pred_boxes.tensor.cpu().numpy()[0][2],
#       panoptic_seg for x in range(len(panoptic_seg["panoptic_seg"]))))]
#df = pd.DataFrame(result, columns = ['class-id','class','score','x-min','y-min','x-max','y-max'])
#print(df)

bounding_boxes = output["instances"]._fields["pred_boxes"].tensor.cpu().numpy()
img_copy = image
for i in range(len(bounding_boxes)):
  left_pt = tuple(bounding_boxes[i][0:2])
  right_pt = tuple(bounding_boxes[i][2:4])
  #cv2.putText(img_copy, f'{i}', left_pt, cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
  cv2.putText(img_copy, f'{i}', (int(left_pt[0]),int(left_pt[1])), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
  #cv2.rectangle(img_copy,left_pt,right_pt,(0,0,155),1)
  cv2.rectangle(img_copy,(int(left_pt[0]),int(left_pt[1])),(int(right_pt[0]),int(right_pt[1])),(0,0,155),1)

cv2.imshow('test_08', img_copy)
cv2.waitKey(0)