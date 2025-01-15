##########################################################################################
# 1. Import packages
##########################################################################################
import torch
import torch.nn as nn

##########################################################################################
# 2. Building an RNN model
##########################################################################################
# Example of a multilayer RNN model with two recurrent layers (num_layers = 2)
# Also included: a non-recurrent fully connected layer (fc) as output layer

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        # Initialize the parent class (nn.Module)
        super().__init__()
        self.rnn = nn.RNN(input_size, 
                          hidden_size, 
                          num_layers=2, 
                          batch_first=True)
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        _, hidden = self.rnn(x)
        # Remember: the hidden tensor has shape (num_layers, batch_size, hidden_size)
        # num_layers = -1 refers to the last layer
        out = hidden[-1, :, :]
        out = self.fc(out)
        return out