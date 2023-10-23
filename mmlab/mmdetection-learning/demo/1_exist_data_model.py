# -*-coding:utf-8 -*-

"""
# File       : 1_exist_data_model.py
# Time       ：2023/2/21 17:00
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from mmdet.apis import init_detector, inference_detector,show_result_pyplot
import mmcv

config_file = "/home/bocheng/source_code/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
checkpoint_file = "/home/bocheng/tmp/mmdet_pth/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

model = init_detector(config_file, checkpoint_file, device="cuda:0")

img = "../../data/Bing - Domestic donkey.jpg"

result = inference_detector(model, img)
show_result_pyplot(model, img, result)