import random

import cv2
#from detectron2.data.catalog import DatasetCatalog
#from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

#データセット登録
#register_coco_instances("tomato", {}, "C:\\Users\\yoshi\\detectron2\\tests\\Tomato\\coco-1697077268.6923862.json", "C:\\Users\\yoshi\\detectron2\\tests\\tomato")
#register_coco_instances("tomato", {}, "./tests/Tomato/coco-1697077268.6923862.json", "./tests/tomato")  #ok
register_coco_instances("tomato", {}, "./tests/Tomato/coco-1697096505.0947952.json", "./tests/tomato")  #ok


#カタログ取得
tmt_metadata = MetadataCatalog.get("tomato")    #メタデータ
dataset_dicts = DatasetCatalog.get("tomato")    #推論結果とクラスラベルの紐づけ

#アノテーションを表示してみる
for d in random.sample(dataset_dicts, 6):
  img = cv2.imread(d["file_name"])
  visualizer = Visualizer(img[:, :, ::-1], metadata=tmt_metadata, scale=1.0)
  vis = visualizer.draw_dataset_dict(d) #RGB形式で画像を受け取るため変換
  #cv2.imshow(d["file_name"], vis.get_image()[:, :, ::-1])
  #cv2.waitKey(0)

#########################################################################################
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo

cfg = get_cfg()
#cfg.OUTPUT_DIR = './output'
cfg.CUDA = 'cuda:0'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.DATASETS.TRAIN = ("tomato",)    #"tomato"データセットに学習
cfg.DATASETS.TEST = ()              #検証用データセット
cfg.DATALOADER.NUM_WORKERS = 2      #データローダーの数
cfg.SOLVER.IMS_PER_BATCH = 2    # 
cfg.SOLVER.BASE_LR = 0.00025    # pick a good LR
cfg.SOLVER.MAX_ITER = 300       # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []           # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 1クラスのみ
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)

if __name__ == '__main__':
    trainer.train()

#訓練結果出力
#   出力フォルダ    C:\Users\yoshi\detectron2\output
#   出力ファイル    model_final.pth