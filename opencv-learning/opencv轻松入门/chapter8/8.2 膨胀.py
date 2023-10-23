# -*-coding:utf-8 -*-

"""
# File       : 8.2 膨胀.py
# Time       ：2023/3/29 14:48
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
import numpy as np

img = np.zeros((5, 5), np.uint8)
img[2:3, 1:4] = 1
kernel = np.ones((3, 1), np.uint8)
dilation = cv2.dilate(img, kernel)
print("img=\n", img)
print("kernel=\n", kernel)
print("dilation\n", dilation)