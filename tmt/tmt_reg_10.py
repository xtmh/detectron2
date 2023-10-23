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
register_coco_instances("tomato", {}, "./tmt/Tomato/Tomato.json", "./tmt/tomato")  #ok

#カタログ取得
tmt_metadata = MetadataCatalog.get("tomato")
dataset_dicts = DatasetCatalog.get("tomato")

#アノテーションを表示してみる
for d in random.sample(dataset_dicts, 6):
  img = cv2.imread(d["file_name"])
  visualizer = Visualizer(img[:, :, ::-1], metadata=tmt_metadata, scale=1.0)
  vis = visualizer.draw_dataset_dict(d) #RGB形式で画像を受け取るため変換
  cv2.imshow(d["file_name"], vis.get_image()[:, :, ::-1])
  cv2.waitKey(0)