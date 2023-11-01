"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-30 12:06:23
LastEditors: chengbo
LastEditTime: 2023-10-30 12:06:30
"""
import os

os.environ["XDG_CACHE_HOME"] = "/data/sshadmin/bocheng/.cache"
# os.environ["HF_CACHE_HOME"] = "/data/sshadmin/bocheng/huggingface"
# os.environ["HF_DATASETS_CACHE"] = "/data/sshadmin/bocheng/huggingface/datasets"
from datasets import load_dataset, DownloadConfig
import datasets


dataset = load_dataset(
    "Graphcore/vqa",
    download_config=DownloadConfig(resume_download=True),
    split="validation[:200]",
)
print(dataset[0])

model_checkpoint = "dandelin/vilt-b32-mlm"
dataset = dataset.remove_columns(["question_type", "question_id", "answer_type"])

from PIL import Image

image = Image.open(dataset[0]["image_id"])
# image.save("./res.jpg")
import itertools

labels = [item["ids"] for item in dataset["label"]]
flatten_labels = list(itertools.chain(*labels))
unique_labels = list(set(flatten_labels))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
print(label2id)


def replace_ids(inputs):
    inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
    return inputs


dataset = dataset.map(replace_ids)
flat_dataset = dataset.flatten()
print(flat_dataset.features)

from transformers import ViltProcessor

processor = ViltProcessor.from_pretrained(model_checkpoint)

import torch


def preprocess_data(examples):
    image_paths = examples["image_id"]
    images = [Image.open(image_path) for image_path in image_paths]
    texts = examples["question"]
    encoding = processor(
        images, texts, padding="max_length", truncation=True, return_tensors="pt"
    )
    for k, v in encoding.items():
        encoding[k] = v.squeeze()
    targets = []
    for labels, scores in zip(examples["label.ids"], examples["label.weights"]):
        target = torch.zeros(len(id2label))
        for label, score in zip(labels, scores):
            target[label] = score
        targets.append(target)
    encoding["labels"] = targets
    return encoding


processed_dataset = flat_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=[
        "question",
        # "question_type",
        # "question_id",
        "image_id",
        # "answer_type",
        "label.ids",
        "label.weights",
    ],
)
print(processed_dataset)

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()
from transformers import ViltForQuestionAnswering

model = ViltForQuestionAnswering.from_pretrained(
    model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id
)
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=4,
    num_train_epochs=20,
    save_steps=200,
    logging_steps=50,
    learning_rate=5e-5,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=processed_dataset,
    tokenizer=processor,
)
trainer.train()
