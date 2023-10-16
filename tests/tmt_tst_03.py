
import os
import cv2
from detectron2.config import get_cfg
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine.defaults import DefaultPredictor
from detectron2.layers import rotated_boxes
from detectron2.utils.visualizer import ColorMode, Visualizer

ver=2

#データセット登録
#register_coco_instances("tomato", {}, "C:\\Users\\yoshi\\detectron2\\tests\\Tomato\\coco-1697077268.6923862.json", "C:\\Users\\yoshi\\detectron2\\tests\\tomato")
register_coco_instances("tomato", {}, "./tests/Tomato/coco-1697173837.322507.json", "./tests/tomato")  #ok

#カタログ取得
tmt_metadata = MetadataCatalog.get("tomato")
dataset_dicts = DatasetCatalog.get("tomato")

#学習モデル
#Faster R-CNN
#config_file = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
#Instance Segmentation
config_file = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
#Keypoint Detection
#config_file = 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
#Panoptic Segmentation
#config_file = 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'

#検出設定
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_file))  #mask
#cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") #生成されたモデル
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05    #低いほど高感度（雑）
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST  = 0.09     #低いほど重ならない（独立）
predictor = DefaultPredictor(cfg)
#
predictor.cfg

#検出テスト
for name in ["s-IMG_2973","s-IMG_3064", "s-IMG_3065", "s-IMG_3066"]:    
    im = cv2.imread(f"./tests/tomato/test/{name}.jpg")
    outputs = predictor(im)

    #検出域を描画
    v = Visualizer(im[:, :, ::-1],
                   metadata=tmt_metadata,
                   instance_mode=ColorMode.IMAGE_BW,
                   scale=1.0
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # Boxes
    #for box in outputs["instances"].pred_boxes.to('cpu'):
    #    v.draw_box(box) 
    # Masks
    #for mask in outputs["instances"].pred_masks.to('cpu'):
    #    v.draw_soft_mask(mask)
    #num_instances = len(box)
    #for lbl in num_instances:
    #    v.draw_rotated_box_with_label(lbl)
    #v = v.get_output()

    #結果画像表示
    cv2.imshow(f"{ver}_{name}.jpg", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)