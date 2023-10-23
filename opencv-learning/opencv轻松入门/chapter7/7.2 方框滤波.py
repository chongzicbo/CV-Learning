# -*-coding:utf-8 -*-

"""
# File       : 7.2 方框滤波.py
# Time       ：2023/3/28 11:56
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import numpy as np
import sys
import cv2

sys.path.append("../")
from config import img_path
from matplotlib import pyplot as plt

o = cv2.imread(img_path)
r = cv2.boxFilter(o, -1, (2, 2), normalize=0)
plt.imshow(o[..., ::-1])
plt.show()
plt.imshow(r[..., ::-1])
plt.show()
