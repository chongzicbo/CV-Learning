# -*-coding:utf-8 -*-

"""
# File       : 6.3 Ostu处理.py
# Time       ：2023/3/27 9:31
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
import numpy as np
import sys
import cv2
sys.path.append("../")
from config import img_path
from matplotlib import pyplot as plt

img = cv2.imread(img_path, 0)
t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
t2, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(img[..., ::-1])
plt.show()
plt.imshow(thd[..., ::-1])
plt.show()

plt.imshow(otsu[..., ::-1])
plt.show()