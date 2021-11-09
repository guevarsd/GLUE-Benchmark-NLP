####
# To be run first
# Run in console
####

!pip install jiant --no-deps


#%%
#####
# Check if installed correctly
####

import sys
sys.path.insert(0, "/content/jiant")

#If these fail, install the packages they request and try again, as many times as needed

import os

import jiant.utils.python.io as py_io
import jiant.proj.simple.runscript as simple_run
import jiant.scripts.download_data.runscript as downloader

#%%

# See https://github.com/nyu-mll/jiant/blob/master/guides/tasks/supported_tasks.md for supported tasks
TASK_NAME = "mrpc"

# See https://huggingface.co/models for supported models
HF_PRETRAINED_MODEL_NAME = "roberta-base"


#%%
# Remove forward slashes so RUN_NAME can be used as path
MODEL_NAME = HF_PRETRAINED_MODEL_NAME.split("/")[-1]
RUN_NAME = f"simple_{TASK_NAME}_{MODEL_NAME}"
EXP_DIR = "/content/exp"
DATA_DIR = "/content/exp/tasks"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

#%%
downloader.download_data([TASK_NAME], DATA_DIR)

#%%
args = simple_run.RunConfiguration(
    run_name=RUN_NAME,
    exp_dir=EXP_DIR,
    data_dir=DATA_DIR,
    hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
    tasks=TASK_NAME,
    train_batch_size=5,
    num_train_epochs=1,
    seed=1, #Change from the sample code - need to set to a value or the call fails
)
simple_run.run_simple(args)

#%%

args = simple_run.RunConfiguration.from_json_path(os.path.join(EXP_DIR, "runs", RUN_NAME, "simple_run_config.json"))
simple_run.run_simple(args)