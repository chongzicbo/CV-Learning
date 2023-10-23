# -*-coding:utf-8 -*-

"""
# File       : paddlehub_test.py
# Time       ：2023/2/1 9:31
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import paddlehub as hub
import cv2

image_path = "/tmp/tmpmuc51thv/7.png"
ocr = hub.Module(name="chinese_ocr_db_crnn_server", enable_mkldnn=True)
result = ocr.recognize_text(images=[cv2.imread(image_path)],visualization=True,output_dir="/home/bocheng/data/paddleocr/paddlehub/test1")

print(result)
