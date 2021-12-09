import optuna.trial as trial
import os
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import transformers


# List of glue tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "cola"  #done: qqp
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

# Number of logits to output
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

# Create model and attach ForSequenceClassification head
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

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
encoded_dataset['train'] = encoded_dataset['train'].remove_columns(['idx', 'sentence'])
encoded_dataset[validation_key] = encoded_dataset[validation_key].remove_columns(['idx', 'sentence'])

trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

best_run = trainer.hyperparameter_search(n_trials=10, direction='maximize')
print(best_run)
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()
trainer.evaluate()


# {'eval_loss': 0.8725433945655823, 'eval_matthews_correlation': 0.4189486545590709, 'eval_runtime': 2.8864, 'eval_samples_per_second': 361.346, 'eval_steps_per_second': 36.377, 'epoch': 4.0}
#
# 100%|███████████████████████████████████████| 4276/4276 [12:22<00:00,  5.82it/s]
#                                                                                 Saving model checkpoint to xlnet-base-cased-finetuned-cola/checkpoint-4276
# Configuration saved in xlnet-base-cased-finetuned-cola/checkpoint-4276/config.json
# Model weights saved in xlnet-base-cased-finetuned-cola/checkpoint-4276/pytorch_model.bin
# tokenizer config file saved in xlnet-base-cased-finetuned-cola/checkpoint-4276/tokenizer_config.json
# Special tokens file saved in xlnet-base-cased-finetuned-cola/checkpoint-4276/special_tokens_map.json
#
#
# Training completed. Do not forget to share your model on huggingface.co/models =)
#
#
# Loading best model from xlnet-base-cased-finetuned-cola/checkpoint-4276 (score: 0.4189486545590709).
# {'train_runtime': 745.1102, 'train_samples_per_second': 45.905, 'train_steps_per_second': 5.739, 'train_loss': 0.5094481094403174, 'epoch': 4.0}
# 100%|███████████████████████████████████████| 4276/4276 [12:25<00:00,  5.74it/s]
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 10
# 100%|█████████████████████████████████████████| 105/105 [00:02<00:00, 36.79it/s]
