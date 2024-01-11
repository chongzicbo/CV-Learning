# -*-coding:utf-8 -*-

"""
# File       : 8.1 腐蚀.py
# Time       ：2023/3/29 14:20
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
import numpy as np
img=np.zeros((5,5),np.uint8)
img[1:4,1:4]=1
kernel=np.ones((3,1),np.uint8)
erosion=cv2.erode(img,kernel)
print("img=\n",img)
print("kernel=\n",kernel)
print("erosion=\n",erosion)
