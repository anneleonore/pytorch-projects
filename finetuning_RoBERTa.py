##########################################################################################
# 1. Import packages
##########################################################################################
import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, AutoTokenizer, get_scheduler, RobertaForSequenceClassification

print("Packages installed")

#General settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 3

##########################################################################################
# 2. Load data
##########################################################################################
#Load dataset
imdb_dataset = load_dataset("imdb")
print("Dataset loaded")

#Define IMDBDataset class
class IMDBDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("IMDBDataset class defined")

##########################################################################################
# 3. Loading the model
##########################################################################################
#Make sure to import the RobertaForSequenceClassification model from the transformers library
from transformers import RobertaForSequenceClassification
print("RobertaForSequenceClassification imported")

#Initialize pre-trained Roberta model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

#Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

#Moving model to specified device
model.to(DEVICE)

#Initialize the Adam optimizer
optimizer = AdamW(model.parameters(), lr=3e-4)

##########################################################################################
# 4. Split the data
##########################################################################################
#Split data in train and validation dataset
train_dataset = IMDBDataset(imdb_dataset['train'], tokenizer)
val_dataset = IMDBDataset(imdb_dataset['test'], tokenizer)

#Create batches of data for each dataset
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

#Define learning rate scheduler (lr_scheduler)
num_training_steps = NUM_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

##########################################################################################
# 5. Define training (fine-tuning) loop
##########################################################################################
#Define progress bar
progress_bar = tqdm(range(num_training_steps))

#Set the model to training mode
model.train()

#Start the loop
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    i = 1
    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            progress_bar.update(10)
        i += 1 
        
    print(f'loss = {epoch_loss / i}, epoch = {epoch}')

##########################################################################################
# 7. Evaluate the model's performance
##########################################################################################
#Define metric
metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

#Set model to evaluation mode
model.eval()

#Start the loop
for batch in val_loader:
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.no_grad(): #disable gradient computation
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

#Compute the metrics
metric.compute()