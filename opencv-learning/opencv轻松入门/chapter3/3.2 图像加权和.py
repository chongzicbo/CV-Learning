# -*-coding:utf-8 -*-

"""
# File       : 3.2 图像加权和.py
# Time       ：2023/3/20 16:41
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2

import numpy as np

img1 = np.ones((3, 4), dtype=np.uint8) * 100
img2 = np.ones((3, 4), dtype=np.uint8) * 10
gamma = 3
img3 = cv2.addWeighted(img1, 0.6, img2, 5, gamma)
print(img3)