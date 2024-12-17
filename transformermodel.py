##########################################################################################
# I. Building the Transformer Model with PyTorch
##########################################################################################

#Step 1. Importing the libraries and modules
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

#Step 2. Defining the basic building blocks
#A. Multi-head attention
class MultiHeadAttention(nn.Module):
    #I. Initialization
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation

    #II. Scaled dot product attention
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
    
    #III. Splitting heads
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    #IV. Combining heads
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    #V. Forward method (where the computation happens) 
    def forward(self, Q, K, V, mask=None):
        #Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        #Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        #Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

# B. Position-wise feed-forward neural networks
class PositionWiseFeedForward(nn.Module):
    #I. Initialization
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        #Two fully linear layers
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        #One Rectified Linear Unit (ReLU) activation function
        self.relu = nn.ReLU()

    #II. Forward method (where the computation happens)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# C. Positional encoding
class PositionalEncoding(nn.Module):
    #I. Initialization
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        #pe will be the tensor filled with positional encodings (but now zero's)
        pe = torch.zeros(max_seq_length, d_model)
        #position holds the position indices for each position in the sequence
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        #The sine function is applied to all even indices of pe
        pe[:, 0::2] = torch.sin(position * div_term)
        #The cosine function is applied to all uneven indices of pe
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    #II. Forward method
    def forward(self, x):
        #Adding the positional encodings to input x
        return x + self.pe[:, :x.size(1)]

#Step 3. Building the encoder block
class EncoderLayer(nn.Module):
    #I. Initialization
    #d_ff is the dimensionality of the inner layer in the position-wise FFNN
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        #A. Multi-head attention mechanism (configuration!)
        #We are configuring the MultiHeadAttention layer with two arguments (d_model and num_heads)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        #B. Position-wise feed-forward neural network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        #C. Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        #D. Dropout layer
        self.dropout = nn.Dropout(dropout)

    #II. Forward method 
    def forward(self, x, mask):
        #A. Input is passed through the multi-head attention mechanism
        #Now we are passing input data through the MultiHeadAttention layer
        #The input values are: x (query), x (key), x (value), mask
        attn_output = self.self_attn(x, x, x, mask)
        #B. Add (residual connection) and normalize
        x = self.norm1(x + self.dropout(attn_output))
        #C. The output from the previous step is passed through the position-wise FFNN
        ff_output = self.feed_forward(x)
        #D. Add (residual connection) and normalize
        x = self.norm2(x + self.dropout(ff_output))
        return x

#Step 4. Building the decoder block
class DecoderLayer(nn.Module):
    #I. Initialize
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        #A. Configuration of MultiheadAttention layer
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        #B. Configuration of Cross attention layer
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        #C. Configuration of Position wise feed forward layer
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        #D. Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    #II. Forward method 
    def forward(self, x, enc_output, src_mask, tgt_mask):
        #A. Let input data flow through first layer
        attn_output = self.self_attn(x, x, x, tgt_mask)
        #B. Add (residual connection) and normalize
        x = self.norm1(x + self.dropout(attn_output))
        #C. Let output data from previous layer flow through second layer
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        #D. Add (residual connection) and normalize
        x = self.norm2(x + self.dropout(attn_output))
        #E. Let output data from previous layer flow through third layer
        ff_output = self.feed_forward(x)
        #F. Add (residual connection) and normalize
        x = self.norm3(x + self.dropout(ff_output))
        return x


#Step 5. Combining the encoder and decoder layers
class Transformer(nn.Module):
    #I. Initialization
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        #A. Embed input sequence into embedding space
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        #B. Embed output sequence into embedding space
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        #C. Add positional information to the embeddings
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        #D. A list of num_layers encoder layers
        #This is a stack of layers: Each object represents a full encoder layer
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        #E. A list of num_layers decoder layers
        #This is a stack of layers: Each object represents a full decoder layer
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        #F. Final linear layer 
        #Here we convert the output of the decoder into a probability distribution over the target vocabulary
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        #G. Dropout layer to regularize the model
        self.dropout = nn.Dropout(dropout)

    #II. Generate mask
    #This ensures that padding tokens are ignored and that future tokens are not visible during training
    def generate_mask(self, src, tgt):
        #A. Create a mask to ignore padding tokens in the input sequences
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        #B. Create a mask to prevent the decoder from attending to future tokens
        #i) Retrieve the length of the target sentence
        #tgt is a tensor with (typically) shape (batch_size, seq_length, embedding_dim)
        seq_length = tgt.size(1)
        #ii) Create triangular matrix with shape (1, seq_length, seq_length), invert (1 - X), and convert to Boolean (.bool)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        #iii) Combine nopeak_mask with existing padding mask (tgt_mask)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    #III. Forward method
    def forward(self, src, tgt):
        #A. Generate the masks for the source (src_mask) and target (tgt_mask) sequences
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        #B. Embed source sequence (also use positional_encoding and apply dropout)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        #C. Embed target sequence (also use positional_encoding and apply dropout)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        #D. Encoder output
        #Initialize enc_output variable with src_embedded
        enc_output = src_embedded
        #Loop that iterates over each encoder layer in the encoder_layers list
        for enc_layer in self.encoder_layers:
            #enc_output is initialized by calling enc_layer with two inputs: enc_output (from the previous layer) and src_mask
            enc_output = enc_layer(enc_output, src_mask)

        #E. Decoder output
        #Initialize dec_output variable with tgt_embedded
        dec_output = tgt_embedded
        #Loop that iterates over each decoder layer in the decoder_layers list
        for dec_layer in self.decoder_layers:
            #dec_output is initialized by calling dec_layer with four inputs:
            #i) dec_output from the previous decoder layer
            #ii) enc_output from the encoder (represents the source sequence)
            #iii) src_mask the mask for the source sequence (helps model ignore padding tokens)
            #iv) tgt_mask for the target sequence (helps model ignore future tokens and padding tokens)
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        #F. Final linear layer (fully connected)
        output = self.fc(dec_output)
        return output
    

##########################################################################################
# II. Prepare data sample
##########################################################################################
#Step 1. Define the hyperparameters
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

#Step 2. Generate random sample
#randint(a, b, size) generates a tensor with random integers uniformly distributed between a and b
#size represents the size of the tensor with the batch_size (64) and the maximum length of a sequence
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

#Step 3. Create a Transformer instance
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
##########################################################################################
# III. Training the Transformer Model
##########################################################################################
#Step 1. Define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

#Step 2. Put the model in training mode
transformer.train()

#Step 3. Train the model
#Loop that iterates over 100 training epochs
for epoch in range(100):
    #Reset the gradients of the model's parameters
    optimizer.zero_grad()
    #Pass the source data and the target data (excluding the last token in each sequence)
    output = transformer(src_data, tgt_data[:, :-1])
    #Calculate the loss between the model's output and target data
    #.contiguous() just ensures that the tensors are stored in contiguous memory locations
    #i) predicted output: output.contiguous().view(-1, tgt_vocab_size)
    #ii) actual output: tgt_data[:, 1:].contiguous().view(-1)
    #We are removing the first token from each sequence
    #view(-1) is reshaping the tensor into a 1D tensor with a single dimension.
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    #Calculate the gradients of the loss with respect to the model's parameters using backpropagation
    loss.backward()
    #Update the model's parameters using the gradients from the previous step
    optimizer.step()
    #Print the epoch number and the loss value
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

##########################################################################################
# IV. Evaluating the Transformer Model
##########################################################################################
#Step 1. Generate random sample validation data
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

#Step 2. Put the model in evaluation mode
transformer.eval()

#Step 3. Evaluate the model
with torch.no_grad(): #disable gradient computation
    #Pass validation source data and target data through the model
    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    #Calculate the loss between the model's predictions and the validation target data (excluding the first token in each sequence)
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    #Print the validation loss
    print(f"Validation Loss: {val_loss.item()}")