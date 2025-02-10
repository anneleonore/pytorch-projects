# I. Import packages
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# II. Create small toy dataset
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

y_train = torch.tensor([0,0,0,1,1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6]
])

y_test = torch.tensor([0,1])

# III. Define a custom Dataset class
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    # Retrieve exactly one data record and the corresponding label
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    # Return the total length of the dataset
    def __len__(self):
        return self.labels.shape[0]

# IV. Create the Dataset objects
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

# V. Instantiate Dataloaders
# Make sure results can be reproduced by setting the manual_seed
torch.manual_seed(123)

train_loader = DataLoader(
    dataset = train_ds,
    batch_size = 2,
    shuffle = True,
    num_workers = 0
    #drop_last = True
)

test_loader = DataLoader(
    dataset = test_ds,
    batch_size = 2,
    shuffle = False,
    num_workers = 0
)

# Iterature over dataloader
for idx, (x,y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)

# VI. Define Neural Network
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
                
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

# VII. Train the model
# Training loop
import torch.nn.functional as F
torch.manual_seed(123)

model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 3

for epoch in range(num_epochs):
    #Set model in training mode
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)
        
        #Loss function
        loss = F.cross_entropy(logits, labels)
        
        optimizer.zero_grad() #zero the gradients of the optimizer
        loss.backward() #backward propagation
        optimizer.step() #update the parameters
    
        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    #Set model in evaluation mode
    model.eval()

# VIII. Use the model on X_train after training
model.eval()
with torch.no_grad():
    outputs = model(X_train)
print(outputs)

# Get class membership probabilities
torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim = 1)
predictions = torch.argmax(probas, dim = 1)

# Check predicted labels for the training set to true training labels
predictions == y_train

# IX. Compute accuracy
def compute_accuracy(model, dataloader):

    # Set model to evaluation mode
    model = model.eval()
    # Initiate variables
    correct = 0.0
    total_examples = 0
    
    for idx, (features, labels) in enumerate(dataloader):
        
        with torch.no_grad():
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

print(compute_accuracy(model, train_loader))
print(compute_accuracy(model, test_loader))

# X. Saving the model
# Save
torch.save(model.state_dict(), "model.pth")

# Restore
model = NeuralNetwork(2, 2)
model.load_state_dict(torch.load("model.pth"))




