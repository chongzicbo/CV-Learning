# -*-coding:utf-8 -*-

"""
# File       : 10.2 Canny函数使用.py
# Time       ：2023/3/30 9:34
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import sys
import cv2

sys.path.append("../")
from config import img_path
from matplotlib import pyplot as plt

o = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
r1 = cv2.Canny(o, 128, 200)
r2 = cv2.Canny(o, 32, 128)
plt.imshow(r1[..., ::-1])
plt.show()
plt.imshow(r2[..., ::-1])
plt.show()
