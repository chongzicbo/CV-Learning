"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-11-01 00:40:24
LastEditors: chengbo
LastEditTime: 2023-11-01 00:47:35
"""
import os

os.environ["XDG_CACHE_HOME"] = "/data/sshadmin/bocheng/.cache"
# os.environ["HF_CACHE_HOME"] = "/data/sshadmin/bocheng/huggingface"
# os.environ["HF_DATASETS_CACHE"] = "/data/sshadmin/bocheng/huggingface/datasets"
from datasets import load_dataset, DownloadConfig
import datasets
from PIL import Image


# 方式一 pipeline
dataset = load_dataset(
    "Graphcore/vqa",
    download_config=DownloadConfig(resume_download=True),
    split="validation[:200]",
)
example = dataset[0]
image = Image.open(example["image_id"])
question = example["question"]
print(question)

from transformers import pipeline

model_output_dir = "./outputs/checkpoint-400"
pipe = pipeline("visual-question-answering", model=model_output_dir)
out = pipe(image, question, top_k=1)
print(out)

# 方式2
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

processor = ViltProcessor.from_pretrained(model_output_dir)
inputs = processor(image, question, return_tensors="pt")
model = ViltForQuestionAnswering.from_pretrained(model_output_dir)
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
idx = logits.argmax(-1).item()
print(model.config.id2label[idx])
