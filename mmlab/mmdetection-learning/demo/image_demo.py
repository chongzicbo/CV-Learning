# -*-coding:utf-8 -*-

"""
# File       : image_demo.py
# Time       ：2023/2/6 14:55
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""

import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector, init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--img", default="/home/bocheng/data/images/kitti_tiny/training/image_2/000001.jpeg",
                        help="Image file")
    parser.add_argument("--config", default="/home/bocheng/tmp/yolov3_mobilenetv2_320_300e_coco.py", help="config file")
    parser.add_argument("--checkpoint",
                        default="/home/bocheng/tmp/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth",
                        help="CheckPoint file")
    parser.add_argument("--out-file", default="./result.jpg", help="Path to output file")
    parser.add_argument("--device", default="cuda:0", help="Path to output file")
    parser.add_argument("--palette", default="coco", choices=["coco", "voc", "citys", "random"],
                        help="Color palette used for visualization")
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    result = inference_detector(model, args.img)
    show_result_pyplot(model, args.img, result, palette=args.palette, score_thr=args.score_thr, out_file=args.out_file)


async def async_main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    show_result_pyplot(model, args.img, result[0], palette=args.palette, score_thr=args.score_thr,
                       out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
