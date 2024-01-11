"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-26 15:22:29
LastEditors: chengbo
LastEditTime: 2023-10-27 10:13:51
"""
"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-26 15:22:29
LastEditors: chengbo
LastEditTime: 2023-10-26 15:22:36
"""
from datasets import load_dataset
from datasets import Dataset
import os
from PIL import Image

base_dir = "/home/bocheng/.cache/huggingface/datasets/downloads/extracted/b36098392eb2e19fc33d6aab25d198289367b27b0db22f8e0a5365c6f71d0b47/ADEChallengeData2016/"


# def my_gen():
#     for i in range(1, 4):
#         yield {"a": i, "b": i + 1}


# def ImageGenerator():
#     for d, sub_d, files in os.walk(base_dir):
#         for filename in files:
#             if
#         break


# ImageGenerator()
# dataset = Dataset.from_generator(my_gen)
# print(dataset[0])

# images = load_dataset("imagefolder", data_dir=os.path.join(base_dir, "images"))
# train_images = images["train"]
# train_images = train_images.remove_columns("label")
# validation_images = images["validation"]
# validation_images = validation_images.remove_columns("label")

# annotations = load_dataset(
#     "imagefolder", data_dir=os.path.join(base_dir, "annotations")
# )

# train_annotations = annotations["train"]
# train_annotations = train_annotations.remove_columns("label").rename_column(
#     "image", "annotation"
# )
# validation_annotations = annotations["validation"]
# validation_annotations = validation_annotations.remove_columns("label").rename_column(
#     "image", "annotation"
# )
# print(train_images)
# train_images.

images_dir = "/data/bocheng/huggingface/data/ADEChallengeData2016/images"
annotations_dir = "/data/bocheng/huggingface/data/ADEChallengeData2016/annotations"


def image_generator(type="training"):
    for image_name in os.listdir(os.path.join(images_dir, type))[:100]:
        image = Image.open(os.path.join(images_dir, type, image_name))
        new_name = image_name.replace("training", "train").replace("jpg", "png")
        label = Image.open(
            os.path.join(
                annotations_dir,
                type,
                new_name,
            )
        )
        yield {"image": image, "labels": label}


dataset = Dataset.from_generator(image_generator)
dataset = dataset.train_test_split(test_size=0.2)
train_ds = dataset["train"]
test_ds = dataset["test"]
