from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
# print(MetadataCatalog.list())
# import torch
# print(torch.cuda.is_available())

import torchvision,torch,detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os,json,cv2,random
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog

img_path="./data/Bing - Domestic donkey.jpg"
img=cv2.imread(img_path)
cv2.imshow("open-cv",img)