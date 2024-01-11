"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-24 14:04:22
LastEditors: chengbo
LastEditTime: 2023-10-25 08:46:34
"""
from dataset_process import DataProcessor
import evaluate
import numpy as np

model_name = "google/vit-base-patch16-224-in21k"  # 使用的模型
model_cache_dir = "/data/bocheng/huggingface/model/"  # 离线下载的模型保存目录
images_dir = "/data/bocheng/cv-data/reverse_image_search"  # 训练数据
dataProcessor = DataProcessor(images_dir, model_name, model_cache_dir)

# 离线加载评估模块，将相应的accuracy.py和accuracy.json放到accuracy目录下即可
accuracy = evaluate.load("/data/bocheng/huggingface/evaluate/accuracy")  # 评估指标


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=len(dataProcessor.labels),
    id2label=dataProcessor.id2label,
    label2id=dataProcessor.label2id,
    cache_dir=model_cache_dir,
)

training_args = TrainingArguments(
    output_dir="./output",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=dataProcessor.data_collator,
    train_dataset=dataProcessor.dataset["train"],
    eval_dataset=dataProcessor.dataset["test"],
    tokenizer=dataProcessor.image_processor,
    compute_metrics=compute_metrics,
)
trainer.train()
