"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-26 11:15:19
LastEditors: chengbo
LastEditTime: 2023-10-26 11:15:29
"""

from transformers import pipeline
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotImageClassification,
    AutoTokenizer,
)

model_name = "/data/bocheng/huggingface/model/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"
model = AutoModelForZeroShotImageClassification.from_pretrained(
    model_name,  # cache_dir=model_cache_dir
)
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
detector = pipeline(
    model=model,
    image_processor=processor,
    tokenizer=tokenizer,
    task="zero-shot-image-classification",
)

from PIL import Image
import requests

url = "/home/bocheng/tmp/pokemon.png"
image = Image.open(url)

predictions = detector(image, candidate_labels=["fox", "bear", "seagull", "owl"])
print(predictions)


candidate_labels = ["tree", "car", "bike", "cat"]
inputs = processor(
    images=image, text=candidate_labels, return_tensors="pt", padding=True
)

import torch

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits_per_image[0]
probs = logits.softmax(dim=-1).numpy()
scores = probs.tolist()
result = [
    {"score": score, "label": candidate_label}
    for score, candidate_label in sorted(
        zip(probs, candidate_labels), key=lambda x: -x[0]
    )
]
print(result)
