# -*-coding:utf-8 -*-

"""
# File       : 2.2 像素处理.py
# Time       ：2023/3/16 9:42
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
import numpy as np
img=np.zeros((8,8),dtype=np.uint8)
print('img=\n',img)
import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()

import numpy as np
import cv2

# -----------蓝色通道值--------------
blue = np.zeros((300, 300, 3), dtype=np.uint8)
blue[:, :, 0] = 255
print("blue=\n", blue)
plt.imshow(blue[...,::-1])
plt.show()
# -----------绿色通道值--------------
green = np.zeros((300, 300, 3), dtype=np.uint8)
green[:, :, 1] = 255
print("green=\n", green)
plt.imshow(green[...,::-1])
plt.show()
# -----------红色通道值--------------
red = np.zeros((300, 300, 3), dtype=np.uint8)
red[:, :, 2] = 255
print("red=\n", red)
plt.imshow(red[...,::-1])
# -----------释放窗口--------------
plt.show()