##########################################################################################
# 1. Import packages
##########################################################################################
import torch
from datasets import load_dataset
from huggingface_hub import HfFolder
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)

print("Packages installed")

##########################################################################################
# 2. Login to HuggingFace
##########################################################################################
#Log in to HuggingFace
from huggingface_hub import login
login()
print("Login successful")

##########################################################################################
# 3. Load data
##########################################################################################
#Define model_id and dataset_id
model_id = "roberta-base"
dataset_id = "ag_news"

#Load dataset
dataset = load_dataset(dataset_id)

print("Dataset loaded")

#repository_id = "achimoraites/roberta-base_ag_news"
##########################################################################################
# 4. Split data
##########################################################################################
#Split data in train and test
train_dataset = dataset['train']
test_dataset = dataset['test'].shard(num_shards=2, index=0)

#Split train_dataset into train and validation sets
val_dataset = dataset['test'].shard(num_shards=2, index=1)

#Initialize Tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

