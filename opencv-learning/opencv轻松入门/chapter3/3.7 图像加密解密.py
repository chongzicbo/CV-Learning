# -*-coding:utf-8 -*-

"""
# File       : 3.7 图像加密解密.py
# Time       ：2023/3/21 11:29
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2

img = cv2.imread('/home/bocheng/data/paddleocr/entailment_as_few-shot_learner.pdf/page-1/0.jpg', 0)
import numpy as np

r, c = img.shape
print(r, c)

key = np.random.randint(0, 256, size=[r, c], dtype=np.uint8)
encryption = cv2.bitwise_xor(img, key)
decryptioin = cv2.bitwise_xor(encryption, key)
from matplotlib import pyplot as plt

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img)

plt.subplot(2, 2, 2)
plt.imshow(key)

plt.subplot(2, 2, 3)
plt.imshow(encryption)

plt.subplot(2, 2, 4)
plt.imshow(decryptioin)

plt.show()
