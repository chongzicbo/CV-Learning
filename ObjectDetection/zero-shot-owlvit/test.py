"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-27 16:47:40
LastEditors: chengbo
LastEditTime: 2023-10-27 16:47:59
"""
from transformers import pipeline, AutoTokenizer, AutoImageProcessor

model_path = "/data/bocheng/huggingface/model/models--google--owlvit-base-patch32/snapshots/8ca8ee912aa922a57e6a89144189080ebc8e852e/"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
image_processor = AutoImageProcessor.from_pretrained(
    pretrained_model_name_or_path=model_path
)
detector = pipeline(
    model=model_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    task="zero-shot-object-detection",
)
import skimage
import numpy as np
from PIL import Image

image = skimage.data.astronaut()
image = Image.fromarray(np.uint8(image)).convert("RGB")
predictions = detector(
    image,
    candidate_labels=["human face", "rocket", "nasa badge", "star-spangled banner"],
)

print(predictions)

from PIL import ImageDraw

draw = ImageDraw.Draw(image)

for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]
    xmin, ymin, xmax, ymax = box.values()
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

image.save("./res.jpg")

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

image = Image.open(
    "/home/bocheng/dev/mylearn/CV-Learning/ObjectDetection/zero-shot-owlvit/zero-sh-obj-detection_3.png"
).convert("RGB")
text_queries = ["hat", "book", "sunglasses", "camera"]
inputs = processor(text=text_queries, images=image, return_tensors="pt")
import torch

with torch.no_grad():
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=0.1, target_sizes=target_sizes
    )[0]
draw = ImageDraw.Draw(image)
scores = results["scores"].tolist()
labels = results["labels"].tolist()
boxes = results["boxes"].tolist()
for box, score, label in zip(boxes, scores, labels):
    xmin, ymin, xmax, ymax = box
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{text_queries[label]}: {round(score,2)}", fill="white")

image.save("./res1.jpg")
