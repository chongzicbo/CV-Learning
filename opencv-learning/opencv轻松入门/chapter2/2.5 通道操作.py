# -*-coding:utf-8 -*-

"""
# File       : 2.5 通道操作.py
# Time       ：2023/3/16 11:35
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
img=cv2.imread('/home/bocheng/data/paddleocr/entailment_as_few-shot_learner.pdf/page-1/0.jpg')
from matplotlib import pyplot as plt
plt.imshow(img[...,::-1])
plt.show()

b=img[:,:,0]
g=img[:,:,1]
r=img[:,:,2]
plt.imshow(b)
plt.show()
plt.imshow(g)
plt.show()
plt.imshow(r)
plt.show()

b,g,r=cv2.split(img)
b = cv2.split(img)[0]
g = cv2.split(img)[1]
r = cv2.split(img)[2]
bgr = cv2.merge([b, g, r]) #合并通道

import cv2

lena = cv2.imread("lenacolor.png")
b, g, r = cv2.split(lena)
bgr = cv2.merge([b, g, r])
rgb = cv2.merge([r, g, b])
cv2.imshow("lena", lena)
cv2.imshow("bgr", bgr)
cv2.imshow("rgb", rgb)
cv2.waitKey()
cv2.destroyAllWindows()