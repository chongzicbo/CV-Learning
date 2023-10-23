# -*-coding:utf-8 -*-

"""
# File       : async_benchmark.py
# Time       ：2023/2/21 17:28
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector,show_result_pyplot
from mmdet.utils.contextmanagers import concurrent

async def main():
    config_file = "/home/bocheng/source_code/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
    checkpoint_file = "/home/bocheng/tmp/mmdet_pth/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    device="cuda:0"

    model=init_detector(config_file,checkpoint=checkpoint_file,device=device)

    streamqueue=asyncio.Queue()

    streamqueue_size=3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    img = "../../data/Bing - Domestic donkey.jpg"
    async with concurrent(streamqueue):
        result=await async_inference_detector(model,img)

    # model.show_result(img,result)
    show_result_pyplot(model,img,result)

asyncio.run(main())