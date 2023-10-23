"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-20 10:52:16
LastEditors: chengbo
LastEditTime: 2023-10-20 10:53:28
"""
import datasets
from datasets import load_dataset

config = datasets.DownloadConfig(resume_download=True, max_retries=100)
cppe5 = load_dataset(
    path="/data/bocheng/huggingface/data/cppe-5",
    # download_config=config,
)
print(cppe5)
print(cppe5["train"][0])
import numpy as np
import os
from PIL import Image, ImageDraw

image = cppe5["train"][0]["image"]
annotations = cppe5["train"][0]["objects"]
draw = ImageDraw.Draw(image)
categories = cppe5["train"].features["objects"].feature["category"].names
print(categories)
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

for i in range(len(annotations["id"])):
    box = annotations["bbox"][i]
    class_idx = annotations["category"][i]
    x, y, w, h = tuple(box)
    draw.rectangle((x, y, x + w, y + w), outline="red", width=1)
    draw.text((x, y), id2label[class_idx], fill="white")


# image.save("./res.jpg")
remove_idx = [590, 821, 822, 875, 876, 878, 879]
keep = [i for i in range(len(cppe5["train"])) if i not in remove_idx]
cppe5["train"] = cppe5["train"].select(keep)

from transformers import AutoImageProcessor

checkpoint = "/data/bocheng/huggingface/model/detr-resnet-50"
checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(
    checkpoint, cache_dir="/data/bocheng/huggingface/model/"
)

import albumentations
import numpy as np
import torch

import albumentations
import numpy as np
import torch

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)


def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


# transforming a batch
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(
            image=image, bboxes=objects["bbox"], category=objects["category"]
        )

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


cppe5["train"] = cppe5["train"].with_transform(transform_aug_ann)
# print(cppe5["train"][15])


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


from transformers import AutoModelForObjectDetection, DetrForObjectDetection, DetrConfig

# config = DetrConfig.from_json_file(
#     "/data/bocheng/huggingface/model/detr-resnet-50/config.json"
# )

# checkpoint = "facebook/detr-resnet-50"
model = DetrForObjectDetection.from_pretrained(
    pretrained_model_name_or_path=checkpoint,  # os.path.join(checkpoint, "pytorch_model.bin"),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    cache_dir="/data/bocheng/huggingface/model/"
    # force_download=True
    # config=config,
)
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="detr-resnet-50_finetuned_cppe5",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    fp16=True,
    save_steps=200,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
)
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=cppe5["train"],
    tokenizer=image_processor,
)

trainer.train()
