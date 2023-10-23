"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-08-31 14:47:56
LastEditors: chengbo
LastEditTime: 2023-10-11 15:32:09
"""
"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-08-31 14:47:56
LastEditors: chengbo
LastEditTime: 2023-10-11 14:24:36
"""
from ultralytics import YOLO

# 目标分割
# model = YOLO("yolov8n-seg.yaml")
# model = YOLO("yolov8n-seg.pt")
# model = YOLO("yolov8n-seg.yaml").load(
#     "yolov8n.pt"
# )  # build from YAML and transfer weights
# model.train(
#     data="/home/bocheng/dev/mylearn/CV-Learning/ultralytics-learning/yolov8-seg/coco128-seg.yaml",
#     epochs=100,
#     imgsz=640,
# )

# 目标检测
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco128.yaml", epochs=10, imgsz=640)
# result = model.val()
result = model("https://ultralytics.com/images/bus.jpg")
print(result)
