"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-24 15:35:40
LastEditors: chengbo
LastEditTime: 2023-10-25 08:48:01
"""
"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-24 15:35:40
LastEditors: chengbo
LastEditTime: 2023-10-24 21:01:59

"""
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from datasets import load_dataset
from transformers import logging
from transformers import AutoImageProcessor
from transformers import DefaultDataCollator


class DataProcessor(object):
    def __init__(self, images_dir, model_name, model_cache_dir) -> None:
        mydataset = load_dataset("imagefolder", data_dir=images_dir)  # 训练数据图片所在文件夹
        self.labels = mydataset["train"].features["label"].names
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name, cache_dir=model_cache_dir
        )
        self.get_id2label()
        self.get_transform()
        self.dataset = mydataset.with_transform(self.transforms)

        self.data_collator = DefaultDataCollator()

    def get_id2label(self):
        label2id, id2label = dict(), dict()
        for i, label in enumerate(self.labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
        self.label2id = label2id
        self.id2label = id2label

    def transforms(self, examples):
        examples["pixel_values"] = [
            self._transforms(img.convert("RGB")) for img in examples["image"]
        ]
        del examples["image"]
        return examples

    def get_transform(self):
        normalize = Normalize(
            mean=self.image_processor.image_mean, std=self.image_processor.image_std
        )
        size = (
            self.image_processor.size["shortest_edge"]
            if "shortest_edge" in self.image_processor.size
            else (
                self.image_processor.size["height"],
                self.image_processor.size["width"],
            )
        )
        _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
        self._transforms = _transforms


if __name__ == "__main__":
    checkpoint = "google/vit-base-patch16-224-in21k"
    cache_dir = "/data/bocheng/huggingface/model/"
    images_dir = "/data/bocheng/cv-data/reverse_image_search"
    dataset = MyDataset(
        images_dir=images_dir, checkpoint=checkpoint, model_cache_dir=cache_dir
    )
    for d in dataset.dataset["train"]:
        print(d)
