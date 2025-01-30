##########################################################################################
# 1. Import packages
##########################################################################################
import matplotlib.pyplot as plt
import numpy as np
import torch

##########################################################################################
# 2. Create a dataset using NumPy and plot
##########################################################################################
X_train = np.arange(10, dtype = 'float32').reshape((10,1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0],
                   dtype = 'float32')

plt.plot(X_train, y_train, 'o', markersize = 10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

##########################################################################################
# 3. Standardize the features and convert to PyTorch tensors
##########################################################################################
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

#Normalize X_train 
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)

#Convert NumPy array to PyTorch tensors
X_train_norm = torch.from_numpy(X_train_norm)
y_train = torch.from_numpy(y_train).float()

##########################################################################################
# 4. Create Dataset and DataLoader
##########################################################################################
#Create PyTorch Dataset for the training set
train_ds = TensorDataset(X_train_norm, y_train)

#Create DataLoader to iterate through individual elements of Dataset
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

##########################################################################################
# 5. Define linear regression model
##########################################################################################
torch.manual_seed(1)

#Initialize weight (single element with random number with mean 0 and std dev 1)
weight = torch.randn(1)
#requires_grad_() > PyTorch will automatically compute the gradients of the tensor during the backward pass
weight.requires_grad_()
#Initialize bias with single element 0
bias = torch.zeros(1, requires_grad = True)

#Define the model (@ is used for matrix multiplication)
def model(xb):
    return xb @ weight + bias

##########################################################################################
# 6. Define loss function (MSE = mean squared error )
##########################################################################################
def loss_fn(input, target):
    return (input-target).pow(2).mean()

##########################################################################################
# 7. Set model parameters
##########################################################################################
learning_rate = 0.001
num_epochs = 200
log_epochs = 10

##########################################################################################
# 8. Training the model
##########################################################################################
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        #Forward pass
        pred = model(x_batch)
        #Calculate the loss
        loss = loss_fn(pred, y_batch)
        #Backward pass
        loss.backward()

        #Update model parameters
        with torch.no_grad(): #disable gradient tracking
            #gradient * learning rate > subtract that from current parameter
            weight -= weight.grad * learning_rate
            bias -= bias.grad * learning_rate
            #Resetting gradients
            weight.grad.zero_()
            bias.grad.zero_()

    #Print statement
    if epoch % log_epochs==0:
        print(f'Epoch {epoch}  Loss {loss.item():.4f}')

#Print final parameters (weight + bias)
print('Final Parameters:', weight.item(), bias.item())

##########################################################################################
# 9. Run model on test data
##########################################################################################
#Create test dataset
#np.linspace requires specifying the number of samples (num parameter = 100 here)
X_test = np.linspace(0, 9, num=100, dtype='float32').reshape(-1, 1)
print(X_test)

#Normalize test dataset
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)

#Convert test dataset to PyTorch tensor
X_test_norm = torch.from_numpy(X_test_norm)

#Making predictions using model()
y_pred = model(X_test_norm).detach().numpy()

##########################################################################################
# 10. Plot the training and test data
##########################################################################################
#Create a figure
fig = plt.figure(figsize=(13, 5))

#Adding a subplot
ax = fig.add_subplot(1, 2, 1)

#Plot training data
plt.plot(X_train_norm, y_train, 'o', markersize=10)

#Plot test data
plt.plot(X_test_norm, y_pred, '--', lw=3)

#Adding a legend
plt.legend(['Training examples', 'Linear Reg.'], fontsize=15)

#Setting axis labels
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)

#Setting tick label sizes
ax.tick_params(axis='both', which='major', labelsize=15)

plt.show()

##########################################################################################
# 11. Model training via torch.nn and torch.optim modules
##########################################################################################
#We are now not manually writing out the loss function and gradient updates
#Instead, we are using the functions provided by torch.nn and torch.optim

import torch.nn as nn

#Define model parameters
input_size = 1
output_size = 1

#Define model (nn.Linear)
model = nn.Linear(input_size, output_size)

#Define loss function (nn.MSELoss)
loss_fn = nn.MSELoss(reduction='mean')

#Define optimizer (torch.optim.SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Define training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:

        #Forward pass
        pred = model(x_batch)[:, 0] 

        #Calculate loss
        loss = loss_fn(pred, y_batch)

        #Backward pass (compute gradients)
        loss.backward()

        #Update parameters using gradients
        optimizer.step()

        #Reset the gradients to zero
        optimizer.zero_grad()
        
    if epoch % log_epochs==0:
        print(f'Epoch {epoch}  Loss {loss.item():.4f}')

print('Final Parameters:', model.weight.item(), model.bias.item())