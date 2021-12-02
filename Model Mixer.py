#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import torch
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import transformers
import os

gc.collect()
torch.cuda.empty_cache()


# In[2]:



# List of glue tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

#List of glue keys
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

#Select task
task = "cola"  #cola, mrpc
batch_size = 10 #10 normally, 8 for qnli

# Load dataset based on task variable
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

#Collect sentence keys and labels
sentence1_key, sentence2_key = task_to_keys[task]

# Number of logits to output
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2


# ## Load Deberta

# In[3]:


###  Tokenizing Section  ####

#Load model
model_checkpoint = "microsoft/deberta-v3-small"

# Create tokenizer for respective model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, truncation=True, model_max_length=512)

def tokenizer_func(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True,)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True,)

# tokenize sentence(s)
encoded_dataset = dataset.map(tokenizer_func, batched=True)

model_checkpoint = "deberta-v3-small_baseline_cola/"

###  Model Section  ####

# Create model and attach ForSequenceClassification head
model_deberta = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

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
    model_deberta,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.evaluate()


# In[5]:


### Collect Predictions  ###

prediction_deberta = trainer.predict(encoded_dataset['test'])


# In[6]:


prediction_deberta


# In[7]:


## Clear the Cache
gc.collect()
torch.cuda.empty_cache()


# ## Load Electra

# In[10]:


###  Tokenizing Section  ####

#Load model
model_checkpoint = "google/electra-small-discriminator"

# Create tokenizer for respective model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, truncation=True, model_max_length=512)

def tokenizer_func(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True,)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True,)

# tokenize sentence(s)
encoded_dataset = dataset.map(tokenizer_func, batched=True)

model_checkpoint = "electra-small-discriminator-finetuned-cola/"

###  Model Section  ####

# Create model and attach ForSequenceClassification head
model_electra = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

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
    model_electra,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.evaluate()


# In[11]:


### Collect Predictions  ###
## Clear the Cache
gc.collect()
torch.cuda.empty_cache()
prediction_electra = trainer.predict(encoded_dataset['test'])


# In[ ]:


## Clear the Cache
gc.collect()
torch.cuda.empty_cache()


# In[ ]:


trainer.prediction_loop()


# In[ ]:


print('done')


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:




