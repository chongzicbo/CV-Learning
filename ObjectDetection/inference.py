from transformers import pipeline, AutoModelForObjectDetection, AutoImageProcessor
import requests
from PIL import Image, ImageDraw
import torch

url = "/home/bocheng/dev/mylearn/CV-Learning/ObjectDetection/cppe5/1002.png"
model_path = "./detr-resnet-50_finetuned_cppe5/checkpoint-1200"
image = Image.open(url)

# 1.使用pipeline接口进行推理
# image_processor = AutoImageProcessor.from_pretrained(
#     "./detr-resnet-50_finetuned_cppe5/checkpoint-1200"
# )
# model = AutoModelForObjectDetection.from_pretrained(
#     "./detr-resnet-50_finetuned_cppe5/checkpoint-1200"
# )
# obj_detector = pipeline(
#     "object-detection", model=model, image_processor=image_processor
# )
# out = obj_detector(image)
# for x in out:
#     print(x)
# out.save("./out.jpg")


image_processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForObjectDetection.from_pretrained(model_path)

with torch.no_grad():
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=0.2, target_sizes=target_sizes
    )[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="red", width=1)
    draw.text((x, y), model.config.id2label[label.item()], fill="white")

image.save("./out.jpg")
