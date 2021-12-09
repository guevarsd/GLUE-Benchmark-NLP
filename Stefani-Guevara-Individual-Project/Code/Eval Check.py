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


# In[3]:


###  Tokenizing Section  ####

#Load model
model_checkpoint = "xlnet-base-cased"

# Create tokenizer for respective model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, truncation=True, model_max_length=512)

def tokenizer_func(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True,)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True,)



# ## Load Deberta

# In[4]:


collector=[]

for task in GLUE_TASKS:

    #Select task
    batch_size = 10 #10 normally

    # Load dataset based on task variable
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)



    #Collect sentence keys and labels
    sentence1_key, sentence2_key = task_to_keys[task]

    # tokenize sentence(s)
    encoded_dataset = dataset.map(tokenizer_func, batched=True)


    # Number of logits to output
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

    #Insert the model checkpoint you want to test
    model_checkpoint = "xlnet-base-cased"
    #model_checkpoint = "deberta-v3-small_baseline_"+actual_task+"/"
    #model_checkpoint = "deberta-v3-small_tuned_"+actual_task+"/"
    
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

    score = trainer.evaluate()
    print(f'{actual_task}: {score}\n\n')
    collector.append([actual_task, metric_name, score])


# In[6]:


print(collector)

#
#
# ssh://ubuntu@3.89.36.198:22/usr/bin/python3 -u "/home/ubuntu/NLP/xlnet_bases/Eval Check.py"
# Downloading: 100%|█████████████████████████████| 760/760 [00:00<00:00, 1.11MB/s]
# Downloading: 100%|███████████████████████████| 798k/798k [00:00<00:00, 56.7MB/s]
# Downloading: 100%|█████████████████████████| 1.38M/1.38M [00:00<00:00, 61.0MB/s]
# Downloading: 28.8kB [00:00, 19.6MB/s]
# Downloading: 28.7kB [00:00, 22.6MB/s]
# Downloading and preparing dataset glue/cola (download: 368.14 KiB, generated: 596.73 KiB, post-processed: Unknown size, total: 964.86 KiB) to /home/ubuntu/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...
# Downloading: 100%|███████████████████████████| 377k/377k [00:00<00:00, 1.05MB/s]
# Dataset glue downloaded and prepared to /home/ubuntu/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.
# Downloading: 5.78kB [00:00, 5.63MB/s]
# 100%|█████████████████████████████████████████████| 9/9 [00:00<00:00, 57.71ba/s]
# 100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 100.31ba/s]
# 100%|█████████████████████████████████████████████| 2/2 [00:00<00:00, 99.75ba/s]
# Downloading: 100%|███████████████████████████| 467M/467M [00:06<00:00, 75.1MB/s]
# Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.weight', 'lm_loss.bias']
# - This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.weight', 'logits_proj.bias', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the evaluation set  don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: idx, sentence.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 10
# 100%|█████████████████████████████████████████| 105/105 [00:02<00:00, 35.40it/s]
# cola: {'eval_loss': 0.8014326691627502, 'eval_matthews_correlation': -0.00938426846530466, 'eval_runtime': 3.0173, 'eval_samples_per_second': 345.669, 'eval_steps_per_second': 34.799}
#
#
# Downloading and preparing dataset glue/mnli (download: 298.29 MiB, generated: 78.65 MiB, post-processed: Unknown size, total: 376.95 MiB) to /home/ubuntu/.cache/huggingface/datasets/glue/mnli/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...
# Downloading: 100%|███████████████████████████| 313M/313M [00:16<00:00, 18.5MB/s]
# Dataset glue downloaded and prepared to /home/ubuntu/.cache/huggingface/datasets/glue/mnli/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.
# 100%|█████████████████████████████████████████| 393/393 [00:17<00:00, 22.63ba/s]
# 100%|███████████████████████████████████████████| 10/10 [00:00<00:00, 20.63ba/s]
# 100%|███████████████████████████████████████████| 10/10 [00:00<00:00, 20.42ba/s]
# 100%|███████████████████████████████████████████| 10/10 [00:00<00:00, 25.97ba/s]
# 100%|███████████████████████████████████████████| 10/10 [00:00<00:00, 20.11ba/s]
# loading configuration file https://huggingface.co/xlnet-base-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/06bdb0f5882dbb833618c81c3b4c996a0c79422fa2c95ffea3827f92fc2dba6b.da982e2e596ec73828dbae86525a1870e513bd63aae5a2dc773ccc840ac5c346
# Model config XLNetConfig {
#   "architectures": [
#     "XLNetLMHeadModel"
#   ],
#   "attn_type": "bi",
#   "bi_data": false,
#   "bos_token_id": 1,
#   "clamp_len": -1,
#   "d_head": 64,
#   "d_inner": 3072,
#   "d_model": 768,
#   "dropout": 0.1,
#   "end_n_top": 5,
#   "eos_token_id": 2,
#   "ff_activation": "gelu",
#   "id2label": {
#     "0": "LABEL_0",
#     "1": "LABEL_1",
#     "2": "LABEL_2"
#   },
#   "initializer_range": 0.02,
#   "label2id": {
#     "LABEL_0": 0,
#     "LABEL_1": 1,
#     "LABEL_2": 2
#   },
#   "layer_norm_eps": 1e-12,
#   "mem_len": null,
#   "model_type": "xlnet",
#   "n_head": 12,
#   "n_layer": 12,
#   "pad_token_id": 5,
#   "reuse_len": null,
#   "same_length": false,
#   "start_n_top": 5,
#   "summary_activation": "tanh",
#   "summary_last_dropout": 0.1,
#   "summary_type": "last",
#   "summary_use_proj": true,
#   "task_specific_params": {
#     "text-generation": {
#       "do_sample": true,
#       "max_length": 250
#     }
#   },
#   "transformers_version": "4.9.2",
#   "untie_r": true,
#   "use_mems_eval": true,
#   "use_mems_train": false,
#   "vocab_size": 32000
# }
#
# loading weights file https://huggingface.co/xlnet-base-cased/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/9461853998373b0b2f8ef8011a13b62a2c5f540b2c535ef3ea46ed8a062b16a9.3e214f11a50e9e03eb47535b58522fc3cc11ac67c120a9450f6276de151af987
# Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.weight', 'lm_loss.bias']
# - This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.weight', 'logits_proj.bias', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# PyTorch: setting up devices
# The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
# The following columns in the evaluation set  don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: premise, idx, hypothesis.
# ***** Running Evaluation *****
#   Num examples = 9815
#   Batch size = 10
# 100%|█████████████████████████████████████████| 982/982 [01:30<00:00, 10.85it/s]
# mnli: {'eval_loss': 1.1359641551971436, 'eval_accuracy': 0.3142129393785023, 'eval_runtime': 90.6555, 'eval_samples_per_second': 108.267, 'eval_steps_per_second': 10.832}
#
#
# Reusing dataset glue (/home/ubuntu/.cache/huggingface/datasets/glue/mnli/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)
# Loading cached processed dataset at /home/ubuntu/.cache/huggingface/datasets/glue/mnli/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad/cache-c90c0f36be20463e.arrow
# Loading cached processed dataset at /home/ubuntu/.cache/huggingface/datasets/glue/mnli/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad/cache-ebfe7386f8bdc20f.arrow
# Loading cached processed dataset at /home/ubuntu/.cache/huggingface/datasets/glue/mnli/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad/cache-3bd0c0ebac9c5c00.arrow
# Loading cached processed dataset at /home/ubuntu/.cache/huggingface/datasets/glue/mnli/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad/cache-065cd696360b2b64.arrow
# Loading cached processed dataset at /home/ubuntu/.cache/huggingface/datasets/glue/mnli/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad/cache-16c5e7ba07b6dc3f.arrow
# loading configuration file https://huggingface.co/xlnet-base-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/06bdb0f5882dbb833618c81c3b4c996a0c79422fa2c95ffea3827f92fc2dba6b.da982e2e596ec73828dbae86525a1870e513bd63aae5a2dc773ccc840ac5c346
# Model config XLNetConfig {
#   "architectures": [
#     "XLNetLMHeadModel"
#   ],
#   "attn_type": "bi",
#   "bi_data": false,
#   "bos_token_id": 1,
#   "clamp_len": -1,
#   "d_head": 64,
#   "d_inner": 3072,
#   "d_model": 768,
#   "dropout": 0.1,
#   "end_n_top": 5,
#   "eos_token_id": 2,
#   "ff_activation": "gelu",
#   "id2label": {
#     "0": "LABEL_0",
#     "1": "LABEL_1",
#     "2": "LABEL_2"
#   },
#   "initializer_range": 0.02,
#   "label2id": {
#     "LABEL_0": 0,
#     "LABEL_1": 1,
#     "LABEL_2": 2
#   },
#   "layer_norm_eps": 1e-12,
#   "mem_len": null,
#   "model_type": "xlnet",
#   "n_head": 12,
#   "n_layer": 12,
#   "pad_token_id": 5,
#   "reuse_len": null,
#   "same_length": false,
#   "start_n_top": 5,
#   "summary_activation": "tanh",
#   "summary_last_dropout": 0.1,
#   "summary_type": "last",
#   "summary_use_proj": true,
#   "task_specific_params": {
#     "text-generation": {
#       "do_sample": true,
#       "max_length": 250
#     }
#   },
#   "transformers_version": "4.9.2",
#   "untie_r": true,
#   "use_mems_eval": true,
#   "use_mems_train": false,
#   "vocab_size": 32000
# }
#
# loading weights file https://huggingface.co/xlnet-base-cased/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/9461853998373b0b2f8ef8011a13b62a2c5f540b2c535ef3ea46ed8a062b16a9.3e214f11a50e9e03eb47535b58522fc3cc11ac67c120a9450f6276de151af987
# Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.weight', 'lm_loss.bias']
# - This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.weight', 'logits_proj.bias', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# PyTorch: setting up devices
# The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
# The following columns in the evaluation set  don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: premise, idx, hypothesis.
# ***** Running Evaluation *****
#   Num examples = 9832
#   Batch size = 10
# 100%|█████████████████████████████████████████| 984/984 [01:31<00:00, 10.81it/s]
# mnli: {'eval_loss': 1.134922742843628, 'eval_accuracy': 0.3109235150528885, 'eval_runtime': 91.3389, 'eval_samples_per_second': 107.643, 'eval_steps_per_second': 10.773}
#
#
# Downloading and preparing dataset glue/mrpc (download: 1.43 MiB, generated: 1.43 MiB, post-processed: Unknown size, total: 2.85 MiB) to /home/ubuntu/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...
# Downloading: 6.22kB [00:00, 6.63MB/s]
# Downloading: 1.05MB [00:00, 2.09MB/s]
# Downloading: 441kB [00:00, 1.13MB/s]
# Dataset glue downloaded and prepared to /home/ubuntu/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.
# 100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 20.77ba/s]
# 100%|█████████████████████████████████████████████| 1/1 [00:00<00:00, 41.49ba/s]
# 100%|█████████████████████████████████████████████| 2/2 [00:00<00:00, 21.46ba/s]
# loading configuration file https://huggingface.co/xlnet-base-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/06bdb0f5882dbb833618c81c3b4c996a0c79422fa2c95ffea3827f92fc2dba6b.da982e2e596ec73828dbae86525a1870e513bd63aae5a2dc773ccc840ac5c346
# Model config XLNetConfig {
#   "architectures": [
#     "XLNetLMHeadModel"
#   ],
#   "attn_type": "bi",
#   "bi_data": false,
#   "bos_token_id": 1,
#   "clamp_len": -1,
#   "d_head": 64,
#   "d_inner": 3072,
#   "d_model": 768,
#   "dropout": 0.1,
#   "end_n_top": 5,
#   "eos_token_id": 2,
#   "ff_activation": "gelu",
#   "initializer_range": 0.02,
#   "layer_norm_eps": 1e-12,
#   "mem_len": null,
#   "model_type": "xlnet",
#   "n_head": 12,
#   "n_layer": 12,
#   "pad_token_id": 5,
#   "reuse_len": null,
#   "same_length": false,
#   "start_n_top": 5,
#   "summary_activation": "tanh",
#   "summary_last_dropout": 0.1,
#   "summary_type": "last",
#   "summary_use_proj": true,
#   "task_specific_params": {
#     "text-generation": {
#       "do_sample": true,
#       "max_length": 250
#     }
#   },
#   "transformers_version": "4.9.2",
#   "untie_r": true,
#   "use_mems_eval": true,
#   "use_mems_train": false,
#   "vocab_size": 32000
# }
#
# loading weights file https://huggingface.co/xlnet-base-cased/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/9461853998373b0b2f8ef8011a13b62a2c5f540b2c535ef3ea46ed8a062b16a9.3e214f11a50e9e03eb47535b58522fc3cc11ac67c120a9450f6276de151af987
# Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.weight', 'lm_loss.bias']
# - This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.weight', 'logits_proj.bias', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# PyTorch: setting up devices
# The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
# The following columns in the evaluation set  don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: sentence2, idx, sentence1.
# ***** Running Evaluation *****
#   Num examples = 408
#   Batch size = 10
# 100%|███████████████████████████████████████████| 41/41 [00:04<00:00, 10.11it/s]
# mrpc: {'eval_loss': 0.6302381753921509, 'eval_accuracy': 0.6862745098039216, 'eval_f1': 0.8128654970760235, 'eval_runtime': 4.2327, 'eval_samples_per_second': 96.392, 'eval_steps_per_second': 9.686}
#
#
# Downloading and preparing dataset glue/qnli (download: 10.14 MiB, generated: 27.11 MiB, post-processed: Unknown size, total: 37.24 MiB) to /home/ubuntu/.cache/huggingface/datasets/glue/qnli/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...
# Downloading: 100%|█████████████████████████| 10.6M/10.6M [00:01<00:00, 10.1MB/s]
# Dataset glue downloaded and prepared to /home/ubuntu/.cache/huggingface/datasets/glue/qnli/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.
# 100%|█████████████████████████████████████████| 105/105 [00:05<00:00, 18.54ba/s]
# 100%|█████████████████████████████████████████████| 6/6 [00:00<00:00, 21.96ba/s]
# 100%|█████████████████████████████████████████████| 6/6 [00:00<00:00, 21.98ba/s]
# loading configuration file https://huggingface.co/xlnet-base-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/06bdb0f5882dbb833618c81c3b4c996a0c79422fa2c95ffea3827f92fc2dba6b.da982e2e596ec73828dbae86525a1870e513bd63aae5a2dc773ccc840ac5c346
# Model config XLNetConfig {
#   "architectures": [
#     "XLNetLMHeadModel"
#   ],
#   "attn_type": "bi",
#   "bi_data": false,
#   "bos_token_id": 1,
#   "clamp_len": -1,
#   "d_head": 64,
#   "d_inner": 3072,
#   "d_model": 768,
#   "dropout": 0.1,
#   "end_n_top": 5,
#   "eos_token_id": 2,
#   "ff_activation": "gelu",
#   "initializer_range": 0.02,
#   "layer_norm_eps": 1e-12,
#   "mem_len": null,
#   "model_type": "xlnet",
#   "n_head": 12,
#   "n_layer": 12,
#   "pad_token_id": 5,
#   "reuse_len": null,
#   "same_length": false,
#   "start_n_top": 5,
#   "summary_activation": "tanh",
#   "summary_last_dropout": 0.1,
#   "summary_type": "last",
#   "summary_use_proj": true,
#   "task_specific_params": {
#     "text-generation": {
#       "do_sample": true,
#       "max_length": 250
#     }
#   },
#   "transformers_version": "4.9.2",
#   "untie_r": true,
#   "use_mems_eval": true,
#   "use_mems_train": false,
#   "vocab_size": 32000
# }
#
# loading weights file https://huggingface.co/xlnet-base-cased/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/9461853998373b0b2f8ef8011a13b62a2c5f540b2c535ef3ea46ed8a062b16a9.3e214f11a50e9e03eb47535b58522fc3cc11ac67c120a9450f6276de151af987
# Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.weight', 'lm_loss.bias']
# - This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.weight', 'logits_proj.bias', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# PyTorch: setting up devices
# The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
# The following columns in the evaluation set  don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: question, idx, sentence.
# ***** Running Evaluation *****
#   Num examples = 5463
#   Batch size = 10
# 100%|█████████████████████████████████████████| 547/547 [01:01<00:00,  8.95it/s]
# qnli: {'eval_loss': 0.717593252658844, 'eval_accuracy': 0.49478308621636463, 'eval_runtime': 61.2659, 'eval_samples_per_second': 89.169, 'eval_steps_per_second': 8.928}
#
#
# Downloading and preparing dataset glue/qqp (download: 39.76 MiB, generated: 106.55 MiB, post-processed: Unknown size, total: 146.32 MiB) to /home/ubuntu/.cache/huggingface/datasets/glue/qqp/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...
# Downloading: 100%|█████████████████████████| 41.7M/41.7M [00:02<00:00, 15.3MB/s]
# Dataset glue downloaded and prepared to /home/ubuntu/.cache/huggingface/datasets/glue/qqp/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.
# 100%|█████████████████████████████████████████| 364/364 [00:13<00:00, 26.85ba/s]
# 100%|███████████████████████████████████████████| 41/41 [00:01<00:00, 25.51ba/s]
# 100%|█████████████████████████████████████████| 391/391 [00:14<00:00, 26.48ba/s]
# loading configuration file https://huggingface.co/xlnet-base-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/06bdb0f5882dbb833618c81c3b4c996a0c79422fa2c95ffea3827f92fc2dba6b.da982e2e596ec73828dbae86525a1870e513bd63aae5a2dc773ccc840ac5c346
# Model config XLNetConfig {
#   "architectures": [
#     "XLNetLMHeadModel"
#   ],
#   "attn_type": "bi",
#   "bi_data": false,
#   "bos_token_id": 1,
#   "clamp_len": -1,
#   "d_head": 64,
#   "d_inner": 3072,
#   "d_model": 768,
#   "dropout": 0.1,
#   "end_n_top": 5,
#   "eos_token_id": 2,
#   "ff_activation": "gelu",
#   "initializer_range": 0.02,
#   "layer_norm_eps": 1e-12,
#   "mem_len": null,
#   "model_type": "xlnet",
#   "n_head": 12,
#   "n_layer": 12,
#   "pad_token_id": 5,
#   "reuse_len": null,
#   "same_length": false,
#   "start_n_top": 5,
#   "summary_activation": "tanh",
#   "summary_last_dropout": 0.1,
#   "summary_type": "last",
#   "summary_use_proj": true,
#   "task_specific_params": {
#     "text-generation": {
#       "do_sample": true,
#       "max_length": 250
#     }
#   },
#   "transformers_version": "4.9.2",
#   "untie_r": true,
#   "use_mems_eval": true,
#   "use_mems_train": false,
#   "vocab_size": 32000
# }
#
# loading weights file https://huggingface.co/xlnet-base-cased/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/9461853998373b0b2f8ef8011a13b62a2c5f540b2c535ef3ea46ed8a062b16a9.3e214f11a50e9e03eb47535b58522fc3cc11ac67c120a9450f6276de151af987
# Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.weight', 'lm_loss.bias']
# - This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.weight', 'logits_proj.bias', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# PyTorch: setting up devices
# The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
# The following columns in the evaluation set  don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: idx, question2, question1.
# ***** Running Evaluation *****
#   Num examples = 40430
#   Batch size = 10
# 100%|███████████████████████████████████████| 4043/4043 [04:16<00:00, 15.79it/s]
# qqp: {'eval_loss': 0.8187657594680786, 'eval_accuracy': 0.37175364828097945, 'eval_f1': 0.5393376618665893, 'eval_runtime': 256.2103, 'eval_samples_per_second': 157.8, 'eval_steps_per_second': 15.78}
#
#
# Downloading and preparing dataset glue/rte (download: 680.81 KiB, generated: 1.83 MiB, post-processed: Unknown size, total: 2.49 MiB) to /home/ubuntu/.cache/huggingface/datasets/glue/rte/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...
# Downloading: 100%|███████████████████████████| 697k/697k [00:00<00:00, 1.48MB/s]
# Dataset glue downloaded and prepared to /home/ubuntu/.cache/huggingface/datasets/glue/rte/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.
# 100%|█████████████████████████████████████████████| 3/3 [00:00<00:00, 18.01ba/s]
# 100%|█████████████████████████████████████████████| 1/1 [00:00<00:00, 48.30ba/s]
# 100%|█████████████████████████████████████████████| 3/3 [00:00<00:00, 10.71ba/s]
# loading configuration file https://huggingface.co/xlnet-base-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/06bdb0f5882dbb833618c81c3b4c996a0c79422fa2c95ffea3827f92fc2dba6b.da982e2e596ec73828dbae86525a1870e513bd63aae5a2dc773ccc840ac5c346
# Model config XLNetConfig {
#   "architectures": [
#     "XLNetLMHeadModel"
#   ],
#   "attn_type": "bi",
#   "bi_data": false,
#   "bos_token_id": 1,
#   "clamp_len": -1,
#   "d_head": 64,
#   "d_inner": 3072,
#   "d_model": 768,
#   "dropout": 0.1,
#   "end_n_top": 5,
#   "eos_token_id": 2,
#   "ff_activation": "gelu",
#   "initializer_range": 0.02,
#   "layer_norm_eps": 1e-12,
#   "mem_len": null,
#   "model_type": "xlnet",
#   "n_head": 12,
#   "n_layer": 12,
#   "pad_token_id": 5,
#   "reuse_len": null,
#   "same_length": false,
#   "start_n_top": 5,
#   "summary_activation": "tanh",
#   "summary_last_dropout": 0.1,
#   "summary_type": "last",
#   "summary_use_proj": true,
#   "task_specific_params": {
#     "text-generation": {
#       "do_sample": true,
#       "max_length": 250
#     }
#   },
#   "transformers_version": "4.9.2",
#   "untie_r": true,
#   "use_mems_eval": true,
#   "use_mems_train": false,
#   "vocab_size": 32000
# }
#
# loading weights file https://huggingface.co/xlnet-base-cased/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/9461853998373b0b2f8ef8011a13b62a2c5f540b2c535ef3ea46ed8a062b16a9.3e214f11a50e9e03eb47535b58522fc3cc11ac67c120a9450f6276de151af987
# Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.weight', 'lm_loss.bias']
# - This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.weight', 'logits_proj.bias', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# PyTorch: setting up devices
# The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
# The following columns in the evaluation set  don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: sentence2, idx, sentence1.
# ***** Running Evaluation *****
#   Num examples = 277
#   Batch size = 10
# 100%|███████████████████████████████████████████| 28/28 [00:06<00:00,  4.24it/s]
# rte: {'eval_loss': 0.7321630716323853, 'eval_accuracy': 0.4584837545126354, 'eval_runtime': 6.8907, 'eval_samples_per_second': 40.199, 'eval_steps_per_second': 4.063}
#
#
# Downloading and preparing dataset glue/sst2 (download: 7.09 MiB, generated: 4.81 MiB, post-processed: Unknown size, total: 11.90 MiB) to /home/ubuntu/.cache/huggingface/datasets/glue/sst2/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...
# Downloading: 100%|█████████████████████████| 7.44M/7.44M [00:00<00:00, 8.42MB/s]
# Dataset glue downloaded and prepared to /home/ubuntu/.cache/huggingface/datasets/glue/sst2/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.
# 100%|███████████████████████████████████████████| 68/68 [00:01<00:00, 40.59ba/s]
# 100%|█████████████████████████████████████████████| 1/1 [00:00<00:00, 37.12ba/s]
# 100%|█████████████████████████████████████████████| 2/2 [00:00<00:00, 36.94ba/s]
# loading configuration file https://huggingface.co/xlnet-base-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/06bdb0f5882dbb833618c81c3b4c996a0c79422fa2c95ffea3827f92fc2dba6b.da982e2e596ec73828dbae86525a1870e513bd63aae5a2dc773ccc840ac5c346
# Model config XLNetConfig {
#   "architectures": [
#     "XLNetLMHeadModel"
#   ],
#   "attn_type": "bi",
#   "bi_data": false,
#   "bos_token_id": 1,
#   "clamp_len": -1,
#   "d_head": 64,
#   "d_inner": 3072,
#   "d_model": 768,
#   "dropout": 0.1,
#   "end_n_top": 5,
#   "eos_token_id": 2,
#   "ff_activation": "gelu",
#   "initializer_range": 0.02,
#   "layer_norm_eps": 1e-12,
#   "mem_len": null,
#   "model_type": "xlnet",
#   "n_head": 12,
#   "n_layer": 12,
#   "pad_token_id": 5,
#   "reuse_len": null,
#   "same_length": false,
#   "start_n_top": 5,
#   "summary_activation": "tanh",
#   "summary_last_dropout": 0.1,
#   "summary_type": "last",
#   "summary_use_proj": true,
#   "task_specific_params": {
#     "text-generation": {
#       "do_sample": true,
#       "max_length": 250
#     }
#   },
#   "transformers_version": "4.9.2",
#   "untie_r": true,
#   "use_mems_eval": true,
#   "use_mems_train": false,
#   "vocab_size": 32000
# }
#
# loading weights file https://huggingface.co/xlnet-base-cased/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/9461853998373b0b2f8ef8011a13b62a2c5f540b2c535ef3ea46ed8a062b16a9.3e214f11a50e9e03eb47535b58522fc3cc11ac67c120a9450f6276de151af987
# Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.weight', 'lm_loss.bias']
# - This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.weight', 'logits_proj.bias', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# PyTorch: setting up devices
# The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
# The following columns in the evaluation set  don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: idx, sentence.
# ***** Running Evaluation *****
#   Num examples = 872
#   Batch size = 10
# 100%|███████████████████████████████████████████| 88/88 [00:04<00:00, 18.86it/s]
# sst2: {'eval_loss': 0.7162917852401733, 'eval_accuracy': 0.5057339449541285, 'eval_runtime': 4.7152, 'eval_samples_per_second': 184.932, 'eval_steps_per_second': 18.663}
#
#
# Downloading and preparing dataset glue/stsb (download: 784.05 KiB, generated: 1.09 MiB, post-processed: Unknown size, total: 1.86 MiB) to /home/ubuntu/.cache/huggingface/datasets/glue/stsb/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...
# Downloading: 100%|███████████████████████████| 803k/803k [00:00<00:00, 1.68MB/s]
# Dataset glue downloaded and prepared to /home/ubuntu/.cache/huggingface/datasets/glue/stsb/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.
# 100%|█████████████████████████████████████████████| 6/6 [00:00<00:00, 33.44ba/s]
# 100%|█████████████████████████████████████████████| 2/2 [00:00<00:00, 38.14ba/s]
# 100%|█████████████████████████████████████████████| 2/2 [00:00<00:00, 44.98ba/s]
# loading configuration file https://huggingface.co/xlnet-base-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/06bdb0f5882dbb833618c81c3b4c996a0c79422fa2c95ffea3827f92fc2dba6b.da982e2e596ec73828dbae86525a1870e513bd63aae5a2dc773ccc840ac5c346
# Model config XLNetConfig {
#   "architectures": [
#     "XLNetLMHeadModel"
#   ],
#   "attn_type": "bi",
#   "bi_data": false,
#   "bos_token_id": 1,
#   "clamp_len": -1,
#   "d_head": 64,
#   "d_inner": 3072,
#   "d_model": 768,
#   "dropout": 0.1,
#   "end_n_top": 5,
#   "eos_token_id": 2,
#   "ff_activation": "gelu",
#   "id2label": {
#     "0": "LABEL_0"
#   },
#   "initializer_range": 0.02,
#   "label2id": {
#     "LABEL_0": 0
#   },
#   "layer_norm_eps": 1e-12,
#   "mem_len": null,
#   "model_type": "xlnet",
#   "n_head": 12,
#   "n_layer": 12,
#   "pad_token_id": 5,
#   "reuse_len": null,
#   "same_length": false,
#   "start_n_top": 5,
#   "summary_activation": "tanh",
#   "summary_last_dropout": 0.1,
#   "summary_type": "last",
#   "summary_use_proj": true,
#   "task_specific_params": {
#     "text-generation": {
#       "do_sample": true,
#       "max_length": 250
#     }
#   },
#   "transformers_version": "4.9.2",
#   "untie_r": true,
#   "use_mems_eval": true,
#   "use_mems_train": false,
#   "vocab_size": 32000
# }
#
# loading weights file https://huggingface.co/xlnet-base-cased/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/9461853998373b0b2f8ef8011a13b62a2c5f540b2c535ef3ea46ed8a062b16a9.3e214f11a50e9e03eb47535b58522fc3cc11ac67c120a9450f6276de151af987
# Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.weight', 'lm_loss.bias']
# - This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.weight', 'logits_proj.bias', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# PyTorch: setting up devices
# The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
# The following columns in the evaluation set  don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: sentence2, idx, sentence1.
# ***** Running Evaluation *****
#   Num examples = 1500
#   Batch size = 10
#  99%|████████████████████████████████████████▋| 149/150 [00:07<00:00, 25.33it/s]Traceback (most recent call last):
#   File "/home/ubuntu/NLP/xlnet_bases/Eval Check.py", line 129, in <module>
#     score = trainer.evaluate()
#   File "/usr/local/lib/python3.8/dist-packages/transformers/trainer.py", line 2041, in evaluate
#     output = eval_loop(
#   File "/usr/local/lib/python3.8/dist-packages/transformers/trainer.py", line 2279, in evaluation_loop
#     metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
#   File "/home/ubuntu/NLP/xlnet_bases/Eval Check.py", line 117, in compute_metrics
#     return metric.compute(predictions=predictions, references=labels)
#   File "/usr/local/lib/python3.8/dist-packages/datasets/metric.py", line 390, in compute
#     self.add_batch(predictions=predictions, references=references)
#   File "/usr/local/lib/python3.8/dist-packages/datasets/metric.py", line 431, in add_batch
#     batch = self.info.features.encode_batch(batch)
#   File "/usr/local/lib/python3.8/dist-packages/datasets/features.py", line 1034, in encode_batch
#     encoded_batch[key] = [encode_nested_example(self[key], obj) for obj in column]
#   File "/usr/local/lib/python3.8/dist-packages/datasets/features.py", line 1034, in <listcomp>
#     encoded_batch[key] = [encode_nested_example(self[key], obj) for obj in column]
#   File "/usr/local/lib/python3.8/dist-packages/datasets/features.py", line 892, in encode_nested_example
#     return schema.encode_example(obj)
#   File "/usr/local/lib/python3.8/dist-packages/datasets/features.py", line 276, in encode_example
#     return float(value)
# TypeError: float() argument must be a string or a number, not 'list'
# 100%|█████████████████████████████████████████| 150/150 [00:07<00:00, 19.07it/s]
#
# Process finished with exit code 1
#
#
#
#
#
