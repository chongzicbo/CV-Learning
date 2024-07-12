"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-24 21:28:24
LastEditors: chengbo
LastEditTime: 2023-10-24 21:28:30
"""
from dataset_process import MyDataset
from transformers import pipeline, AutoModelForImageClassification

checkpoint = "google/vit-base-patch16-224-in21k"
cache_dir = "/data/bocheng/huggingface/model/"
images_dir = "/data/bocheng/cv-data/reverse_image_search"
dataset = MyDataset(
    images_dir=images_dir, checkpoint=checkpoint, model_cache_dir=cache_dir
)
test_dataset = dataset.dataset["test"]
print(test_dataset[0])

checkpoint_dir = "./output/checkpoint-48"
model = AutoModelForImageClassification.from_pretrained(
    checkpoint_dir,
)
test_image_path = "/home/bocheng/data/images/reverse_image_search/test/Afghan_hound/n02088094_4261.JPEG"
classifier = pipeline(
    "image-classification", image_processor=dataset.image_processor, model=model
)
from PIL import Image


print(classifier(Image.open(test_image_path)))
