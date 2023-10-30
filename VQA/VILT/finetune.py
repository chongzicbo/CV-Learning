"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-30 12:06:23
LastEditors: chengbo
LastEditTime: 2023-10-30 12:06:30
"""
from datasets import load_dataset

dataset = load_dataset("Graphcore/vqa", split="validation[:200]")
dataset
