import os

import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import transformers


# List of glue tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "cola"  #done: qqp
actual_task = "mnli" if task == "mnli-mm" else task
batch_size = 10
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

###  Tokenizing Section  ####

#Load model
model_checkpoint = "xlnet-base-cased"

# Create tokenizer for respective model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, truncation=True, model_max_length=512)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]

def tokenizer_func(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True,)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True,)

# tokenize sentence(s)
encoded_dataset = dataset.map(tokenizer_func, batched=True)

model_checkpoint = "xlnet-base-cased-finetuned-cola/checkpoint-4280"

###  Model Section  ####
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
# Create model and attach ForSequenceClassification head
model_xlnet = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Type of metric for given task
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"


args = TrainingArguments(
    f"{model_checkpoint}-finetuned-Testing-{task}",
    evaluation_strategy = "epoch",
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    metric_for_best_model=metric_name,
    eval_accumulation_steps=5
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model_xlnet,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.evaluate()
prediction_xlnet = trainer.predict(encoded_dataset[validation_key])

print(prediction_xlnet)
