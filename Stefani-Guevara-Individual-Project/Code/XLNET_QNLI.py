import os

import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import transformers


# List of glue tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "qnli"  # getting cuda out of memory
model_checkpoint = "xlnet-base-cased"
batch_size = 3

# Load dataset based on task variable
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

# Path to save model weights
PATH = os.getcwd()

# Create tokenizer for respective model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, model_max_length=500, truncation=True)

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
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Number of logits to output
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

# Create model and attach ForSequenceClassification head
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Type of metric for given task
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

args = TrainingArguments(
    f"{model_checkpoint}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
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

# Remove unnecessary columns for training
encoded_dataset['train'] = encoded_dataset['train'].remove_columns(['idx', 'question', 'sentence'])
encoded_dataset[validation_key] = encoded_dataset[validation_key].remove_columns(['idx', 'question', 'sentence'])

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    # data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()


# 2 epoch
# {'eval_loss': 0.5688426494598389, 'eval_accuracy': 0.8779059125022881, 'eval_runtime': 56.7402, 'eval_samples_per_second': 96.281, 'eval_steps_per_second': 32.094, 'epoch': 2.0}
#  67%|█████████████████████▎          | 69830/104745 [3:35:12<1:47:26,  5.42it/s]
# 100%|███████████████████████████████████████| 1821/1821 [00:56<00:00, 31.92it/s]
#                                                                                 Saving model checkpoint to xlnet-base-cased-finetuned-qnli/checkpoint-69830
# Configuration saved in xlnet-base-cased-finetuned-qnli/checkpoint-69830/config.json
# Model weights saved in xlnet-base-cased-finetuned-qnli/checkpoint-69830/pytorch_model.bin
# tokenizer config file saved in xlnet-base-cased-finetuned-qnli/checkpoint-69830/tokenizer_config.json
# Special tokens file saved in xlnet-base-cased-finetuned-qnli/checkpoint-69830/special_tokens_map.json


# ! pip install optuna
# ! pip install ray[tune]

# def model_init():
#     return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
#
#
# trainer = Trainer(
#     model_init=model_init,
#     args=args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset[validation_key],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
# best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
#
# for n, v in best_run.hyperparameters.items():
#     setattr(trainer.args, n, v)
#
# trainer.train()
