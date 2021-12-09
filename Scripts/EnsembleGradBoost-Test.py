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

metric_collector = []


# List of glue tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]


# In[2]:


for task in GLUE_TASKS:
    
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
    batch_size = 10 #10 normally, 8 for qnli
    
    # Load dataset based on task variable
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)
    
    #Collect sentence keys and labels
    sentence1_key, sentence2_key = task_to_keys[task]
    
    # Number of logits to output
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    
    

    ###############################################
    
    #         DEBERTA SECTION
    
    ###############################################
    
    
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

    test_key = "test_mismatched" if task == "mnli-mm" else "test_matched" if task == "mnli" else "test"
    test_data = encoded_dataset[test_key].remove_columns('label')
    
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
            predictions = predictions[:]#, 0]
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
    

    
    ### Collect Predictions  ###
    
    prediction_deberta = trainer.predict(test_data)

    
    ## Clear the Cache
    gc.collect()
    torch.cuda.empty_cache()
    
    
    ###############################################
    
    #         ELECTRA SECTION
    
    ###############################################
    
    
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
    
    test_key = "test_mismatched" if task == "mnli-mm" else "test_matched" if task == "mnli" else "test"
    test_data = encoded_dataset[test_key].remove_columns('label')
    
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


    
    ### Collect Predictions  ###
    ## Clear the Cache
    gc.collect()
    torch.cuda.empty_cache()
    prediction_electra = trainer.predict(test_data)
    

    
    
    ## Clear the Cache
    gc.collect()
    torch.cuda.empty_cache()
    


    ###############################################
    
    #         XLNET SECTION
    
    ###############################################
    
    
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
    
    test_key = "test_mismatched" if task == "mnli-mm" else "test_matched" if task == "mnli" else "test"
    test_data = encoded_dataset[test_key].remove_columns('label')
    
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
    

    
    ### Collect Predictions  ###
    ## Clear the Cache
    gc.collect()
    torch.cuda.empty_cache()
    prediction_xlnet = trainer.predict(test_data)
    

    
    ## Clear the Cache
    gc.collect()
    torch.cuda.empty_cache()
    

    
    

    ###############################################
    
    # Combine Model Predicions to create Input Features
    
    ###############################################
    
    
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
    
    

    
    
    #Combine the dataframes
    df_combine = pd.concat([df_deberta, df_electra, df_xlnet], axis=1)
    df_combine.head()
    
    
    ###############################################
    
    #         ENSEMBLE SECTION
    
    ###############################################
    
    
    # Importing the required packages
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import classification_report
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr

    import joblib

    

    ###
    #Load the Ensemble Method
    search = joblib.load("./Ensemble Models/"+task+"_GradBoost_ensemble2.joblib")
    
    # Predict the values
    y_pred = search.predict(df_combine)
    
    # Print basic Report, then specify for the model
    
    print("\n-------------------------------------\n")
    print('\n\n\n\n')
    print("TASK : ", task, "COMPLETE")
    print('\n\n\n\n')
    print("\n-------------------------------------\n")
    
    #Save predictions
    np.savetxt('./Predictions/gradient_boost_fit1_'+task+'.csv', y_pred, delimiter=',')
    
    


# In[3]:


print('Done')

