# -*-coding:utf-8 -*-

"""
# File       : 3.3 按位逻辑运算.py
# Time       ：2023/3/20 16:48
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
# import cv2
# import numpy as np
#
# a = np.random.randint(0, 255, (5, 5), dtype=np.uint8)
# b = np.zeros((5, 5), dtype=np.uint8)
# b[0:3, 0:3] = 255
# b[4, 4] = 255
# print(a)
# print(b)
# c = cv2.bitwise_and(a, b)
# print("a=\n", a)
# print("b=\n", b)
# print("c=\n", c)

import cv2
import numpy as np

a = cv2.imread("/home/bocheng/data/paddleocr/entailment_as_few-shot_learner.pdf/page-1/0.jpg", 0)
b = np.zeros(a.shape, dtype=np.uint8)
b[100:400, 200:400] = 255
b[100:500, 100:200] = 255
c = cv2.bitwise_and(a, b)


from matplotlib import pyplot as plt
plt.figure()
plt.subplot(3,1,1)
plt.imshow(a)

plt.subplot(3,1,2)
plt.imshow(b)

plt.subplot(3,1,3)
plt.imshow(c)

plt.show()