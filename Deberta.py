#import gc
import torch

#gc.collect()
torch.cuda.empty_cache()


import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import transformers
import os

# List of glue tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "mnli-mm"  #cola, mrpc, mnli-kind of, qnli-kind of, sst2-kind-of, wnli 
model_checkpoint = "microsoft/deberta-v3-small"
batch_size = 4 #10 normally, 8 for qnli

#Need to load deberta-v3-small-finetuned-mnli/run-0/checkpoint-98176

#Verify baseline not already established
path = 'deberta-v3-small_tuned_'+task

if path in os.listdir():
    response = input(f'Tuning for {task} already established. Continue? [y/n]')
    if response.lower() in ['yes', 'y']:
        print('Continuing.')
    else:
        raise ValueError('Stopping Process.')

del os, torch


# Load dataset based on task variable
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

# Create tokenizer for respective model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, model_max_length=100, truncation=True)
#512

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


# select columns with sentences to tokenize based on given task
sentence1_key, sentence2_key = task_to_keys[task]
def tokenizer_func(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

# tokenize sentence(s)
encoded_dataset = dataset.map(tokenizer_func, batched=True)

encoded_dataset=encoded_dataset.remove_columns(column_names=["idx", "premise", "hypothesis"])
#encoded_dataset=encoded_dataset.remove_columns(column_names=["idx", "question", "sentence"])
#encoded_dataset=encoded_dataset.remove_columns(column_names=["idx", "sentence1", "sentence2"])
#encoded_dataset=encoded_dataset.remove_columns(column_names=["idx", "sentence"])


# data_collator = DataCollatorForLanguageModeling( tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Number of logits to output
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

# Create model and attach ForSequenceClassification head
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Type of metric for given task
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
# model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_checkpoint}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    eval_accumulation_steps=5
    # push_to_hub=True,
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
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    # data_collator=data_collator,
    compute_metrics=compute_metrics
)

import optuna
# ! pip install optuna
# ! pip install ray[tune]

def model_init():
     return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

#
trainer = Trainer(
     model_init=model_init,
     args=args,
     train_dataset=encoded_dataset["train"],
     eval_dataset=encoded_dataset[validation_key],
     tokenizer=tokenizer,
     compute_metrics=compute_metrics
)
best_run = trainer.hyperparameter_search(n_trials=5, direction="maximize")
#
for n, v in best_run.hyperparameters.items():
     setattr(trainer.args, n, v)
#
trainer.train()

#Save Model
print(path)
model.save_pretrained(path)
