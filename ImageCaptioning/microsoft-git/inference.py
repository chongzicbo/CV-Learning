"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-26 10:03:21
LastEditors: chengbo
LastEditTime: 2023-10-26 10:05:08
"""
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

url = "/home/bocheng/tmp/pokemon.png"
image = Image.open(url)

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(
    checkpoint, cache_dir="/data/bocheng/huggingface/model/"
)
inputs = processor(images=image, return_tensor="pt")
pixel_values = inputs.pixel_values[0]
input_ids = torch.unsqueeze(torch.tensor(pixel_values), 0)
checkpoint = "/home/bocheng/dev/mylearn/CV-Learning/ImageCaptioning/microsoft-git/output-pokemon/checkpoint-4700"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
generated_ids = model.generate(pixel_values=input_ids, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_token=True)[0]
print(generated_caption)
