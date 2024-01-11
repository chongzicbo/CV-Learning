import os

os.environ["XDG_CACHE_HOME"] = "/data/sshadmin/bocheng/.cache"
model_checkpoint = "microsoft/layoutlmv2-base-uncased"
batch_size = 4
from datasets import load_dataset

dataset = load_dataset("nielsr/docvqa_1200_examples")
print(dataset)
print(dataset["train"].features)
updated_dataset = dataset.map(
    lambda example: {"question": example["query"]["en"]}, remove_columns=["query"]
)
updated_dataset = updated_dataset.map(
    lambda example: {"answer": example["answers"][0]},
    remove_columns=["answer", "answers"],
)
updated_dataset = updated_dataset.filter(
    lambda x: len(x["words"]) + len(x["question"].split()) < 512
)
updated_dataset = updated_dataset.remove_columns("words")
updated_dataset = updated_dataset.remove_columns("bounding_boxes")
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_checkpoint)
image_processor = processor.image_processor


def get_ocr_words_and_boxes(examples):
    images = [image.convert("RGB") for image in examples["image"]]
    encoded_inputs = image_processor(images)

    examples["image"] = encoded_inputs.pixel_values
    examples["words"] = encoded_inputs.words
    examples["boxes"] = encoded_inputs.boxes

    return examples


dataset_with_ocr = updated_dataset.map(
    get_ocr_words_and_boxes, batched=True, batch_size=2
)
tokenizer = processor.tokenizer
