##########################################################################################
# 1. Import packages
##########################################################################################
import sys
print("Python interpreter:", sys.executable)
print("Virtual environment:", sys.prefix)

import gzip
import os
import pandas as pd
import shutil
import time
import torch
import torch.nn.functional as F
import torchtext
import transformers
import requests

print("Packages installed")
##########################################################################################
# 2. General settings
##########################################################################################
#Ensure reproducibility
torch.backends.cudnn.deterministic = True
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 3

##########################################################################################
# 3. Fetch data
##########################################################################################
#This is a single string (with implicit line joining)
url = ("https://github.com/rasbt/"
       "machine-learning-book/raw/"
       "main/ch08/movie_data.csv.gz")
filename = url.split("/")[-1]

#Create file on local disk
with open(filename, "wb") as f:
    r  = requests.get(url)
    f.write(r.content)

#Print the file path
print(f"File saved to: {os.path.abspath(filename)}")

#Open contents of compressed file (f_in) to decompressed file (f_out)
with gzip.open('movie_data.csv.gz', 'rb') as f_in:
    with open('movie_data.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Data successfully fetched")

#Load data into pandas Dataframe
df = pd.read_csv('movie_data.csv')

#Print first three rows to check if everything is correct
print(df.head(3))

##########################################################################################
# 4. Split dataset into training, validation and test set
##########################################################################################
#I. Training set
train_texts = df.iloc[:35000]['review'].values
train_labels = df.iloc[:35000]['sentiment'].values

#II. Validation set
valid_texts = df.iloc[35000:40000]['review'].values
valid_labels = df.iloc[35000:40000]['sentiment'].values

#III. Test set
test_texts = df.iloc[40000:]['review'].values
test_labels = df.iloc[40000:]['sentiment'].values

##########################################################################################
# 5. Tokenize the dataset
##########################################################################################
#Make sure to import the DistilBertTokenizer fast from the transformers library
from transformers import DistilBertTokenizerFast
print("DistilBertTokenizerFast imported")

#Using tokenizer implementation inherited from the pre-trained model class
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

#Remember: The tokenizer function expects a list of strings as input
train_encodings = tokenizer(list(train_texts), truncation = True, padding = True)
valid_encodings = tokenizer(list(valid_texts), truncation = True, padding = True)
test_encodings = tokenizer(list(test_texts), truncation = True, padding = True)

#Double check what the maximum sequence length is (max_length = 512)
print(tokenizer.model_max_length)

##########################################################################################
# 6. Define a class
##########################################################################################
#Define a new class called IMDbDataset that inherits from the PyTorch's Dataset class
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels): #encodings and labels are the parameters passed
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        #Creating a dictionary with key:value pairs
        #idx is an index value that takes on values: 0, 1, 2, 3, etc.
        #val[idx] accesses the value at the current index idx
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        #Creating a new key:value pair with 'labels' as key
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

#Create three instances of IMDbDataset class with different attributes
train_dataset = IMDbDataset(train_encodings, train_labels)
valid_dataset = IMDbDataset(valid_encodings, valid_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

#Create batches of data for each dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 16, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 16, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 16, shuffle = True)

##########################################################################################
# 7. Loading and fine-tunin a pre-trained BERT model
##########################################################################################
#Make sure to import the DistilBertForSequenceClassification model from the transformers library
from transformers import DistilBertForSequenceClassification
print("DistilBertForSequenceClassification imported")

#Initialize pre-trained BERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

#Moving model to specified device
model.to(DEVICE)

#Initialize the Adam optimizer
optim = torch.optim.Adam(model.parameters(), lr=5e-5)

##########################################################################################
# 8. Define function to calculate model accuracy (part of test loop)
##########################################################################################
def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        #Initialize accuracy tracking variables
        correct_pred, num_examples = 0, 0
        #Iterate over the dataloader
        for batch_idx, batch in enumerate(data_loader):
            #Moving input data to specified device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            #Forward pass
            outputs = model(input_ids, attention_mask = attention_mask)
            #Logits extraction
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)
            #Update accuracy tracking variables
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
    #Return accuracy (percentage)
    return correct_pred.float()/num_examples * 100

print("Function compute_accuracy defined")

##########################################################################################
# 9. Define training (fine-tuning) loop
##########################################################################################
start_time = time.time()

print("Start finetuning the model...")

for epoch in range(NUM_EPOCHS):

    #Set the model to training mode
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        #Moving input data to specified device
        input_ids = batch['input_ids'].to(DEVICE)
        #attention_mask is a binary mask that specifies whether tokens are real or padding
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        #Forward pass
        outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
        #Logits extraction
        loss, logits = outputs['loss'], outputs['logits']

        #Backward pass
        optim.zero_grad() #zero the gradients of the parameters
        loss.backward()
        optim.step()

        #Logging
        if not batch_idx % 250:
            print(f'Epoch: {epoch+1:04d}/{NUM_EPOCHS:04d}'
                  f'| Batch'
                  f'{batch_idx:04d}'
                  f'{len(train_loader):04d} |'
                  f'Loss: {loss:.4f}')


    #Set model to evaluation mode
    model.eval()

    with torch.set_grad_enabled(False):
        print(f'Training accuracy: '
              f'{compute_accuracy(model, train_loader, DEVICE):.2f}%'
              f'\nValid accuracy: '
              f'{compute_accuracy(model, valid_loader, DEVICE):.2f}%')
        
        print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
    
    print(f'Total training time: {(time.time() - start_time)/60:.2f} min')
    print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')

