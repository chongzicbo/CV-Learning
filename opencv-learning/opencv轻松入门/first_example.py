# -*-coding:utf-8 -*-

"""
# File       : first_example.py
# Time       ：2023/3/15 16:42
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2

lena = cv2.imread('/home/bocheng/data/paddleocr/entailment_as_few-shot_learner.pdf/page-1/0.jpg')
cv2.nameWindow("lesson")
cv2.imshow("lesson", lena)
