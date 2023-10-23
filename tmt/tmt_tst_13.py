
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
import pathlib
import csv

ver = 12

#データセット登録
#register_coco_instances("tomato", {}, "C:\\Users\\yoshi\\detectron2\\tests\\Tomato\\coco-1697077268.6923862.json", "C:\\Users\\yoshi\\detectron2\\tests\\tomato")
register_coco_instances("tomato", {}, "./tmt/Tomato/Tomato.json", "./tmt/tomato")  #ok

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
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") #生成されたモデル
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05    #低いほど高感度（雑）
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST  = 0.09     #低いほど重ならない（独立）
predictor = DefaultPredictor(cfg)

#検出テスト
indir = "./tmt/tomato"
otdir = "./tmt/tomato/dest/"
header_row = ['FileName', 'leaf','board','tomato','FilePath']
with open(otdir+"dest.csv", "w", newline="", encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header_row)

    #img_list = list(pathlib.Path(indir).glob('**/*.jpg'))  #   全階層対応
    #img_list = list(pathlib.Path(indir).glob('test2/*.jpg'))    #   下位階層のみ対応
    img_list = list(pathlib.Path(indir).glob('test/*.jpg'))    #   下位階層のみ対応
    for i in range(len(img_list)):
        im = cv2.imread(str(img_list[i]))    #, cv2.IMREAD_GRAYSCALE))
        outputs = predictor(im)

        # class毎の検出数出力
        pd = outputs["instances"].pred_classes.to("cpu")
        xnum = pd.bincount()
        if len(xnum) == 3:
            stnum = ("leaf({}),  board({}), tomato({})".format(xnum[0], xnum[1], xnum[2]))
            dnum = (f"{xnum[0]}, {xnum[1]}, {xnum[2]}")
        elif len(xnum) == 2:
            stnum = ("leaf({}),  board({}), tomato(0)".format(xnum[0], xnum[1]))
            dnum = (f"{xnum[0]}, {xnum[1]}, 0")
        elif len(xnum) == 1:
            stnum = ("leaf({}),  board(0), tomato(0)".format(xnum[0]))
            dnum = (f"{xnum[0]}, 0, 0")
        print(stnum)

        #csv出力
        img_name = img_list[i].name
        img_path = img_list[i]
        drow = (f"{img_name}, {dnum}, {img_path}")
        writer.writerow(drow.split(","))
    
        #検出域を描画（Visualizerオブジェクト）
        vlz = Visualizer(im[:, :, ::-1], metadata=tmt_metadata, instance_mode=ColorMode.IMAGE_BW, scale=1.0)
        vlz.draw_text(stnum, (0,0), horizontal_alignment="left")

        bFlag = 1
        if bFlag:
            v = vlz.draw_instance_predictions(outputs["instances"].to("cpu"))   #VisImageを出力
        else:
            # Boxes
            #for box in outputs["instances"].pred_boxes.to('cpu'):
            #    v.draw_box(box, edge_color="b")    #   boxを表示
                #v.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()))  #   座標を表示
 
            # Masks
            for mask in outputs["instances"].pred_masks.to('cpu'):
                vlz.draw_soft_mask(mask)    #   Visualizerにmaskを描画
            v = vlz.get_output()            #   VisImageを出力

        #結果画像表示
        out_path = otdir + img_name
        #cv2.imshow(f"{ver}_{img_name}.jpg", v.get_image()[:, :, ::-1])  #   v : VisImageを表示
        #画像保存
        v.save(str(out_path))
    cv2.waitKey(0)