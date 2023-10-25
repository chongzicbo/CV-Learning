"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-25 13:42:13
LastEditors: chengbo
LastEditTime: 2023-10-25 16:02:20
"""
"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-25 13:42:13
LastEditors: chengbo
LastEditTime: 2023-10-25 13:42:50
"""
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM

from datasets import load_dataset

ds = load_dataset(path="/data/bocheng/huggingface/data/pokemon-blip-captions/")
print(ds)
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]


checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(
    checkpoint, cache_dir="/data/bocheng/huggingface/model/"
)


model = AutoModelForCausalLM.from_pretrained(
    checkpoint, cache_dir="/data/bocheng/huggingface/model/"
)


def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs


train_ds.set_transform(transforms)
test_ds.set_transform(transforms)
from evaluate import load
import torch

wer = load("/data/bocheng/huggingface/evaluate/wer")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}


from transformers import TrainingArguments, Trainer

model_name = checkpoint.split("/")[1]

training_args = TrainingArguments(
    output_dir="output-pokemon",
    learning_rate=5e-5,
    num_train_epochs=50,
    fp16=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=50,
    remove_unused_columns=False,
    label_names=["labels"],
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
trainer.train()
