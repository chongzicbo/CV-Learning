"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-26 14:03:12
LastEditors: chengbo
LastEditTime: 2023-10-27 10:14:26
"""
from datasets import load_dataset
import json
from data_process import train_ds, test_ds

label_id_file = "/data/bocheng/huggingface/data/ade20k-id2label.json"
id2label = json.load(open(label_id_file))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

from transformers import AutoImageProcessor

model_name_or_path = "/data/bocheng/huggingface/model/models--nvidia--mit-b0/snapshots/ed0b85c75627eab6a3c6989627450cf95f115381"
image_processor = AutoImageProcessor.from_pretrained(
    model_name_or_path, reduce_labels=True
)

from torchvision.transforms import ColorJitter

jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)


def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch["image"]]
    labels = example_batch["labels"]
    inputs = image_processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [jitter(x) for x in example_batch["image"]]
    labels = example_batch["labels"]
    inputs = image_processor(images, labels)
    return inputs


train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

import evaluate

metric = evaluate.load("/data/bocheng/huggingface/evaluate/mean_iou/")

import numpy as np
import torch
from torch import nn


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        logits_tensor = logits_tensor.argmax(dim=1)
        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        return metrics


from transformers import AutoModelForSemanticSegmentation, Trainer, TrainingArguments

model = AutoModelForSemanticSegmentation.from_pretrained(
    model_name_or_path, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="outputs",
    learning_rate=6e-5,
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
