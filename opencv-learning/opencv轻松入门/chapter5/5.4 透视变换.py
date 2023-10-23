# -*-coding:utf-8 -*-

"""
# File       : 5.4 透视变换.py
# Time       ：2023/3/24 9:23
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/bocheng/data/paddleocr/entailment_as_few-shot_learner.pdf/page-1/0.jpg')
rows, cols = img.shape[:2]
print(rows, cols)
pts1 = np.float32([[150, 50], [400, 50], [60, 450], [310, 450]])
pts2 = np.float32([[50, 50], [rows - 50, 50], [50, cols - 50], [rows - 50, cols - 50]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (cols, rows))
plt.imshow(img)
plt.show()
plt.imshow(dst)
plt.show()