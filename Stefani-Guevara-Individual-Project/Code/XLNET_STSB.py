import os

import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import transformers


# List of glue tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "stsb"  #done: qqp, cola, rte, sst2
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
    learning_rate=2e-5,
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
# {'eval_loss': 0.5013921856880188, 'eval_pearson': 0.8926391576772031, 'eval_spearmanr': 0.8887598545455353, 'eval_runtime': 7.6953, 'eval_samples_per_second': 194.923, 'eval_steps_per_second': 19.492, 'epoch': 5.0}
# 100%|███████████████████████████████████████| 2875/2875 [14:18<00:00,  3.31it/s]
# 100%|█████████████████████████████████████████| 150/150 [00:07<00:00, 25.21it/s]
#                                                                                 Saving model checkpoint to xlnet-base-cased-finetuned-stsb/checkpoint-2875
# Configuration saved in xlnet-base-cased-finetuned-stsb/checkpoint-2875/config.json
# Model weights saved in xlnet-base-cased-finetuned-stsb/checkpoint-2875/pytorch_model.bin
# tokenizer config file saved in xlnet-base-cased-finetuned-stsb/checkpoint-2875/tokenizer_config.json
# Special tokens file saved in xlnet-base-cased-finetuned-stsb/checkpoint-2875/special_tokens_map.json
#
#
# Training completed. Do not forget to share your model on huggingface.co/models =)
#
#
# Loading best model from xlnet-base-cased-finetuned-stsb/checkpoint-2875 (score: 0.8926391576772031).
# {'train_runtime': 861.1305, 'train_samples_per_second': 33.381, 'train_steps_per_second': 3.339, 'train_loss': 0.49057172161599866, 'epoch': 5.0}
# 100%|███████████████████████████████████████| 2875/2875 [14:21<00:00,  3.34it/s]
# ***** Running Evaluation *****
#   Num examples = 1500
#   Batch size = 10
# 100%|█████████████████████████████████████████| 150/150 [00:07<00:00, 18.96it/s]
# Configuration saved in /home/ubuntu/NLPxlnet-base-cased_baseline_stsb/config.json
# Model weights saved in /home/ubuntu/NLPxlnet-base-cased_baseline_stsb/pytorch_model.bin
