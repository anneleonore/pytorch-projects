####################################################################################
# Import packages
import torch
import torch.nn as nn

# Sample data
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

####################################################################################
# I. Define MultiHeadAttention
####################################################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Shape is (b, num_tokens, d_out)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split the matrix by unrolling d_out to num_heads and head_dim
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Switch num_tokens (1 > 2) and num_heads (2 > 1)
        # Important: We need to calculate attention scores with regard to the  head_dim
        # Dimension: b, num_heads, num_tokens, head_dim
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calculate attn_scores by matrix mulitplication of queries and keys
        # keys.transpose(2,3) dimension: b, num_heads, head_dim, num_tokens
        attn_scores = queries @ keys.transpose(2, 3)

        # Create mask and apply to attn_scores
        # [:num_tokens, :num_tokens] > to make sure mask is applied to full attn_scores matrix
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalize attn_scores to get attn_weights | dim = -1 : Apply softmax operation to last dimension of attn_scores tensor
        # keys.shape[-1] refers to the last dimension of the keys tensor: head_dim
        # head_dim here: Features that a particular attention head is processing
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)

        # Apply dropout layer
        attn_weights = self.dropout(attn_weights)

        # Calculate context vectors and transpose back to original shape
        # Original shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1,2)

        # Combine the outputs of the heads (self.d_out = self.num_heads * self.head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Apply linear projection to context_vec
        context_vec = self.out_proj(context_vec)

        return context_vec
        
print("MultiHeadAttention class defined")
  
####################################################################################
# II. Implement the MultiHeadAttention class
####################################################################################
# Create batch of data
batch = torch.stack((inputs, inputs), dim = 0)
print(batch.shape)

# Set random seed for reproducibility
torch.manual_seed(123)

# Extract the batch size, context length, and input dimension from the batch tensor
batch_size, context_length, d_in = batch.shape

# Set output dimension
d_out = 2

# Initialize a MultiHeadAttention module
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads = 2)

# Compute context_vecs by applying MultiHeadAttention module to batch
context_vecs = mha(batch)

print(context_vecs)