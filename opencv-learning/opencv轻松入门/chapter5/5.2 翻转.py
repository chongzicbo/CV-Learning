# -*-coding:utf-8 -*-

"""
# File       : 5.2 翻转.py
# Time       ：2023/3/24 8:54
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/bocheng/data/paddleocr/entailment_as_few-shot_learner.pdf/page-1/0.jpg')
x = cv2.flip(img, 0)
y = cv2.flip(img, 1)
xy = cv2.flip(img, -1)

plt.imshow(img[...,::-1])
plt.show()
plt.imshow(x[...,::-1])
plt.show()
plt.imshow(y[...,::-1])
plt.show()
plt.imshow(xy[...,::-1])
plt.show()