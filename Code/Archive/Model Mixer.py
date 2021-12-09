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
task = "rte"  #cola, mrpc
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

#model_checkpoint = "deberta-v3-small_baseline_cola/"
model_checkpoint = "deberta-v3-small_baseline_"+actual_task+"/"

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


# In[4]:


### Collect Predictions  ###

prediction_deberta = trainer.predict(encoded_dataset[validation_key])


# In[5]:


prediction_deberta


# In[6]:


## Clear the Cache
gc.collect()
torch.cuda.empty_cache()


# ## Load Electra

# In[7]:


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

#model_checkpoint = "electra-small-discriminator-finetuned-cola/"
#model_checkpoint = "Electra_fintuned_cola/"
model_checkpoint = "Electra_fintuned_"+actual_task+"/"

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


# In[8]:


### Collect Predictions  ###
## Clear the Cache
gc.collect()
torch.cuda.empty_cache()
prediction_electra = trainer.predict(encoded_dataset[validation_key])


# In[9]:


## Clear the Cache
gc.collect()
torch.cuda.empty_cache()


# In[10]:


prediction_electra


# In[11]:


print('done')


# ## Load XLNet

# In[12]:


###  Tokenizing Section  ####

#Load model
model_checkpoint = "xlnet-base-cased"

# Create tokenizer for respective model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, truncation=True, model_max_length=512)

def tokenizer_func(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True,)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True,)

# tokenize sentence(s)
encoded_dataset = dataset.map(tokenizer_func, batched=True)

#model_checkpoint = "electra-small-discriminator-finetuned-cola/"
#model_checkpoint = "Electra_fintuned_cola/"
model_checkpoint = "xlnet-base-cased_baseline_"+actual_task+"/"

###  Model Section  ####

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


# In[13]:


### Collect Predictions  ###
## Clear the Cache
gc.collect()
torch.cuda.empty_cache()
prediction_xlnet = trainer.predict(encoded_dataset[validation_key])


# In[14]:


## Clear the Cache
gc.collect()
torch.cuda.empty_cache()


# In[15]:


prediction_xlnet


# In[ ]:





# # Random Forest

# ## Combine Model Predicions to create Input Features

# In[16]:


import pandas as pd

#Labels
val_labels = prediction_deberta.label_ids


#DeBERTa
df_deberta = pd.DataFrame(prediction_deberta[0])
df_deberta=df_deberta.rename(columns=dict(zip(df_deberta.columns,['deberta_'+str(col) for col in df_deberta.columns])))
print(df_deberta.head(),'\n')


#Electra
df_electra = pd.DataFrame(prediction_electra[0])
df_electra=df_electra.rename(columns=dict(zip(df_electra.columns,['electra_'+str(col) for col in df_electra.columns])))
print(df_electra.head(),'\n')


#XLNet
df_xlnet = pd.DataFrame(prediction_xlnet[0])
df_xlnet=df_xlnet.rename(columns=dict(zip(df_xlnet.columns,['xlnet_'+str(col) for col in df_xlnet.columns])))
print(df_xlnet.head(),'\n')


# In[17]:


#Combine the dataframes
df_combine = pd.concat([df_deberta, df_electra, df_xlnet], axis=1)
df_combine.head()


# In[18]:


# Importing the required packages

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


# In[19]:


# Split the dataset into train and test

X_train, X_test, y_train, y_test = train_test_split(df_combine, val_labels, test_size=0.3, random_state=100)
X_train.head()


# In[20]:


# Perform training with random forest with all columns
# Initialize random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# Perform training
clf.fit(X_train, y_train)

# Predicton on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)


# In[21]:


metric_name


# In[22]:


# Print basic Report, then specify for the model

print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

if metric_name == 'accuracy':
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    
elif metric_name == 'matthews_correlation':
    print("Accuracy : ", matthews_corrcoef(y_test, y_pred) * 100)

elif metric_name == "pearson":
    print("Accuracy : ", matthews_corrcoef(y_test, y_pred) * 100)

else:
    print('ERROR')

print('-------------------')
print("DeBERTa : ", prediction_deberta.metrics['test_'+metric_name]*100)
print("Electra : ", prediction_electra.metrics['test_'+metric_name]*100)
print("XLNet : ", prediction_xlnet.metrics['test_'+metric_name]*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


get_ipython().system('nvidia-smi')


# In[24]:


### How to make the test dataset not crash

#test2 = encoded_dataset['test'].remove_columns(['label'])
#new_column = [0] * len(test2)
#test3 = test2.add_column("label", new_column)

#Should run fine
#prediction_electra = trainer.predict(test3)
#prediction_electra[0]


# In[ ]:




