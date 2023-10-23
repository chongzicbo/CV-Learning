# -*-coding:utf-8 -*-

"""
# File       : 6.1 threshold.py
# Time       ：2023/3/27 8:50
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print("img=\n", img)
print("t=", t)
print("rst=\n", rst)

import sys

sys.path.append("../")
from config import img_path
from matplotlib import pyplot as plt

img = cv2.imread(img_path)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
plt.imshow(img[..., ::-1])
plt.show()

plt.imshow(rst[..., ::-1])
plt.show()

t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
plt.imshow(rst[..., ::-1])
plt.show()

t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
plt.imshow(rst[..., ::-1])
plt.show()


t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
plt.imshow(rst[..., ::-1])
plt.show()