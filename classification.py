##########################################################################################
# 1. Import packages
##########################################################################################
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

##########################################################################################
# 2. Load data
##########################################################################################
iris = load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1./3, random_state = 1
)

##########################################################################################
# 3. Standardize features
##########################################################################################
#Standardize
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_test_norm = (X_test - np.mean(X_test)) / np.std(X_test)

#Transform to PyTorch tensors
X_train_norm = torch.from_numpy(X_train_norm).float()
X_test_norm = torch.from_numpy(X_test_norm).float()
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

##########################################################################################
# 4. Create TensorDataset and DataLoader
##########################################################################################
#Create TensorDataset (pairs of tensors)
train_ds = TensorDataset(X_train_norm, y_train)

#Set manual_seed
torch.manual_seed(1)

#Create DataLoader
train_dl = DataLoader(train_ds, batch_size = 2, shuffle = True)

##########################################################################################
# 5. Define model
##########################################################################################
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        #Importing all methods and attributes from nn.Module 
        super().__init__()
        #Adding two new attributes
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.Sigmoid()(x)
        x = self.layer2(x)
        return x

##########################################################################################
# 6. Create instance of the model
##########################################################################################
#Define input_size, hidden_size and output_size
input_size = X_train_norm.shape[1] #input_size is the number of features
print(f"Input size: {input_size}")
hidden_size = 16
output_size = 3

#Creating an instance of the model
model = Model(input_size, hidden_size, output_size)

##########################################################################################
# 7. Set up for training
##########################################################################################
#Set learning rate
learning_rate = 0.001

#Define loss function
loss_fn = torch.nn.CrossEntropyLoss()
 
#Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##########################################################################################
# 7. Train the model
##########################################################################################
#Defining hyperparameter
num_epochs = 100 #how often we will iterate through the dataset

#Defining history variables (creating a list with repeating 0 num_epochs times)
loss_hist = [0] * num_epochs
accuracy_hist = [0] * num_epochs

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        #Forward pass
        pred = model(x_batch)
        #Calculate loss
        loss = loss_fn(pred, y_batch.long())
        #Backward pass
        loss.backward()
        #Updating model parameters
        optimizer.step()
        #Resetting gradients
        optimizer.zero_grad()

        #Updating loss and accuracy histories
        loss_hist[epoch] += loss.item()*y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist[epoch] += is_correct.sum()
    
    #Normalizing loss and accuracy histories
    loss_hist[epoch] /= len(train_dl.dataset)
    accuracy_hist[epoch] /= len(train_dl.dataset)

##########################################################################################
# 8. Visualization
##########################################################################################
#Create a figure
fig = plt.figure(figsize=(12, 5))
#Add a subplot (1)
ax = fig.add_subplot(1, 2, 1)
#Plotting the loss_hist
ax.plot(loss_hist, lw=3)
#Customizing the loss plot
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

#Adding a subplot (2)
ax = fig.add_subplot(1, 2, 2)
#Plotting the accuracy_hist
ax.plot(accuracy_hist, lw=3)
#Customizing the accuracy plot
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
 
#plt.show()

##########################################################################################
# 9. Evaluate trained model on test data
##########################################################################################
#Run trained model on test dataset
pred_test = model(X_test_norm)

#Print statements
print("X_test_norm shape:", X_test_norm.shape) #[50,4]
print("X_test_norm (first 5 rows):", X_test_norm[:5])
print("pred_test shape:", pred_test.shape) #[50,3]
print("pred_test (first 5 rows):", pred_test[:5])

#Check structure of y_test
print("y_test shape and values:", y_test)

#Calculate correct predictions and compute accuracy
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()
print(f'Test Acc.: {accuracy:.4f}')

##########################################################################################
# 10. Save the model
##########################################################################################
#Define path and save model
path = 'iris_classifier.pt'
torch.save(model,path)

#Reload model
model_new = torch.load(path)

#Verify architecture of model
print(model_new.eval())

#Only save learned parameters
path2 = 'iris_classifier_state.pt'
torch.save(model.state_dict(), path)

#Reload model
model_new = Model(input_size, hidden_size, output_size)
model_new.load_state_dict(torch.load(path2))
