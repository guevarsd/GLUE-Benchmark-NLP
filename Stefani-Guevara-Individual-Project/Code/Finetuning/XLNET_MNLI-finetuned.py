import os
import optuna
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import transformers


# List of glue tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "mnli"  #done: qqp, cola, rte, sst2, stsb, wnli
model_checkpoint = "xlnet-base-cased"
batch_size = 5

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

# Number of logits to output
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

# Create model and attach ForSequenceClassification head
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

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
    num_train_epochs=2,
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
encoded_dataset['train'] = encoded_dataset['train'].remove_columns(['idx', 'premise', 'hypothesis'])
encoded_dataset[validation_key] = encoded_dataset[validation_key].remove_columns(['idx', 'premise', 'hypothesis'])

trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

best_run = trainer.hyperparameter_search(n_trials=10, direction='maximize', backend='optuna')
print(best_run)
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()
trainer.evaluate()

# 
# {'eval_loss': 0.6514620780944824, 'eval_accuracy': 0.8573611818644932, 'eval_runtime': 80.3942, 'eval_samples_per_second': 122.086, 'eval_steps_per_second': 24.417, 'epoch': 2.0}
# 100%|████████████████████████████████| 196352/196352 [10:40:11<00:00,  5.52it/s]
# 100%|███████████████████████████████████████| 1963/1963 [01:20<00:00, 23.67it/s]
#                                                                                 Saving model checkpoint to xlnet-base-cased-finetuned-mnli/run-0/checkpoint-196352
# Configuration saved in xlnet-base-cased-finetuned-mnli/run-0/checkpoint-196352/config.json
# Model weights saved in xlnet-base-cased-finetuned-mnli/run-0/checkpoint-196352/pytorch_model.bin
# tokenizer config file saved in xlnet-base-cased-finetuned-mnli/run-0/checkpoint-196352/tokenizer_config.json
# Special tokens file saved in xlnet-base-cased-finetuned-mnli/run-0/checkpoint-196352/special_tokens_map.json
#
#
# Training completed. Do not forget to share your model on huggingface.co/models =)
#
#
# Loading best model from xlnet-base-cased-finetuned-mnli/run-0/checkpoint-196352 (score: 0.8573611818644932).
# {'train_runtime': 38413.3517, 'train_samples_per_second': 20.446, 'train_steps_per_second': 5.112, 'train_loss': 0.5971469171976639, 'epoch': 2.0}
# 100%|████████████████████████████████| 196352/196352 [10:40:13<00:00,  5.11it/s]
# [I 2021-12-05 01:30:19,556] Trial 0 finished with value: 0.8573611818644932 and parameters: {'learning_rate': 1.958137953203701e-05, 'num_train_epochs': 2, 'seed': 12, 'per_device_train_batch_size': 4}. Best is trial 0 with value: 0.8573611818644932.
