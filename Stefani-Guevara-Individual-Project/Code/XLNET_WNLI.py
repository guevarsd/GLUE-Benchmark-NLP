import os

import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import transformers


# List of glue tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "wnli"  #done: qqp, cola, rte, sst2, stsb
model_checkpoint = "xlnet-base-cased"
batch_size = 10

# Load dataset based on task variable
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

# Path to save model weights
PATH = os.getcwd()

# Create tokenizer for respective model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

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
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
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

# Remove unnecessary columns for training
encoded_dataset['train'] = encoded_dataset['train'].remove_columns(['idx', 'sentence1', 'sentence2'])
encoded_dataset[validation_key] = encoded_dataset[validation_key].remove_columns(['idx', 'sentence1', 'sentence2'])

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

# 5 epochs
# {'eval_loss': 0.6903868913650513, 'eval_accuracy': 0.5633802816901409, 'eval_runtime': 0.6276, 'eval_samples_per_second': 113.131, 'eval_steps_per_second': 12.747, 'epoch': 5.0}
# 100%|█████████████████████████████████████████| 320/320 [01:51<00:00,  3.60it/s]
# 100%|█████████████████████████████████████████████| 8/8 [00:00<00:00, 13.28it/s]
#                                                                                 Saving model checkpoint to xlnet-base-cased-finetuned-wnli/checkpoint-320
# Configuration saved in xlnet-base-cased-finetuned-wnli/checkpoint-320/config.json
# Model weights saved in xlnet-base-cased-finetuned-wnli/checkpoint-320/pytorch_model.bin
# tokenizer config file saved in xlnet-base-cased-finetuned-wnli/checkpoint-320/tokenizer_config.json
# Special tokens file saved in xlnet-base-cased-finetuned-wnli/checkpoint-320/special_tokens_map.json
#
#
# Training completed. Do not forget to share your model on huggingface.co/models =)
#
#
# Loading best model from xlnet-base-cased-finetuned-wnli/checkpoint-64 (score: 0.5633802816901409).
# {'train_runtime': 114.0726, 'train_samples_per_second': 27.833, 'train_steps_per_second': 2.805, 'train_loss': 0.7072928428649903, 'epoch': 5.0}
# 100%|█████████████████████████████████████████| 320/320 [01:54<00:00,  2.81it/s]
# ***** Running Evaluation *****
#   Num examples = 71
#   Batch size = 10
# 100%|█████████████████████████████████████████████| 8/8 [00:00<00:00, 14.81it/s]
# Configuration saved in /home/ubuntu/NLPxlnet-base-cased_baseline_wnli/config.json
# Model weights saved in /home/ubuntu/NLPxlnet-base-cased_baseline_wnli/pytorch_model.bin
