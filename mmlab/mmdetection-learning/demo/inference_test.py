"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-17 11:45:01
LastEditors: chengbo
LastEditTime: 2023-10-17 11:45:08
"""
from mmdet.apis import DetInferencer

# 初始化模型
inferencer = DetInferencer("rtmdet_tiny_8xb32-300e_coco")

# 推理示例图片
inferencer(
    "/home/bocheng/data/images/kitti_tiny/training/image_2/000001.jpeg",
    out_dir="outputs/",
    no_save_pred=False,
    pred_score_thr=0.1,  # 置信度阈值，超过该值则视为检测出的结果
)
