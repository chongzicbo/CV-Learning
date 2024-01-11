# -*-coding:utf-8 -*-

"""
# File       : 5.3 仿射.py
# Time       ：2023/3/24 9:10
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/bocheng/data/paddleocr/entailment_as_few-shot_learner.pdf/page-1/0.jpg')

##平移
height, width = img.shape[:2]
x = 100
y = 200
# M = np.float32([[1, 0, x], [0, 1, y]])
# move = cv2.warpAffine(img, M, (width, height))
# plt.imshow(img[...,::-1])
# plt.show()
# plt.imshow(move[...,::-1])
# plt.show()

## 旋转
M = cv2.getRotationMatrix2D((height / 2, width / 2), 45, 0.6)
print(M)

rotate=cv2.warpAffine(img,M,(width,height))
plt.imshow(rotate[...,::-1])
plt.show()
