# -*-coding:utf-8 -*-

"""
# File       : 3.1 图像加法运算.py
# Time       ：2023/3/20 16:34
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import numpy as np

img1 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
img2 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
print("img1=\n", img1)
print("img2=\n", img2)
print("img1+img2=\n", img1 + img2)

import numpy as np
import cv2

img1 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
img2 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
print("img1=\n", img1)
print("img2=\n", img2)
img3 = cv2.add(img1, img2)
print("cv2.add(img1, img2)=\n", img3)