# -*-coding:utf-8 -*-

"""
# File       : 5.1 缩放.py
# Time       ：2023/3/24 8:46
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
import numpy as np

img = np.ones([2, 4, 3], dtype=np.uint8)
size = img.shape[:2]
rst = cv2.resize(img, size)
print("img.shape=\n", img.shape)
print("img=\n", img)
print("rst.shape=\n", rst.shape)
print("rst=\n", rst)

img = cv2.imread('/home/bocheng/data/paddleocr/entailment_as_few-shot_learner.pdf/page-1/0.jpg')
rows, cols = img.shape[:2]
size = (int(cols * 0.9), int(rows * 0.5))
rst = cv2.resize(img, size)
print("img.shape=", img.shape)
print("rst.shape=", rst.shape)

import cv2

rst = cv2.resize(img, None, fx=2, fy=0.5)
print("img.shape=", img.shape)
print("rst.shape=", rst.shape)