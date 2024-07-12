# -*-coding:utf-8 -*-

"""
# File       : 6.2 自适应阈值处理.py
# Time       ：2023/3/27 9:17
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
import sys

sys.path.append("../")
from config import img_path
from matplotlib import pyplot as plt

img = cv2.imread(img_path, 0)
t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
athdMEAN = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
athdGAUS = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)

plt.imshow(img[..., ::-1])
plt.show()
plt.imshow(thd[..., ::-1])
plt.show()
plt.imshow(athdMEAN[..., ::-1])
plt.show()

plt.imshow(athdGAUS[..., ::-1])
plt.show()
