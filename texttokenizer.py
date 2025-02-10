####################################################################################
# Import packages
import requests
import re
import tiktoken
import torch
import urllib.request
from torch.utils.data import Dataset, DataLoader


####################################################################################
# A. Create simple text tokenizer
####################################################################################
# I. Import text
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")

url_text = requests.get(url)
raw_text = url_text.text

# II. Preprocessing text
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

# III. Create vocabulary
all_tokens= sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) #Adding two special tokens to all_tokens
vocab_size = len(all_tokens)
vocab = {token:integer for integer, token in enumerate(all_tokens)}

# IV. Create tokenizer
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int 
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove spaces before the specified punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# V. Instantiate new tokenizer objects 
tokenizer = SimpleTokenizerV1(vocab)

# VI. Create small example to check if tokenizer works
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

print(tokenizer.encode(text))
print("")
print(tokenizer.decode(tokenizer.encode(text)))

####################################################################################
# B. Bye pair encoding (BPE) tokenizer
####################################################################################
# I. Instantiate BPE tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# II. Create small example
text = ("Hello, do you like tea? <|endoftext|> In the sunlit terrances"
        "of someunknownPlace.")

# Print token IDs
integers = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})
print(integers)

# Convert token IDs back to text using decode method
strings = tokenizer.decode(integers)
print(strings)

# III. Create example with unknown word
text_example = "Akwirw ier"
integers_example = tokenizer.encode(text_example)
print(f"Integers example: {integers_example}")
strings_example = tokenizer.decode(integers_example)
print(strings_example)

####################################################################################
# C. From token IDs to word embeddings
####################################################################################
# I. Small example
input_ids = torch.tensor([2, 3, 5, 1])

# Create vocab_size and output_dim
vocab_size = 6
output_dim = 3

# Instantiate embedding layer
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Print underlying weight matrix of the embedding layer
print(embedding_layer.weight)

# II. Apply to one single token ID to obtain embedding vector
print(embedding_layer(torch.tensor([3])))

# III. Apply to full tensor
print(embedding_layer(input_ids))

####################################################################################
# C. Tokenizer with positional embeddings
####################################################################################
# I. Instantiate token embedding layer
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# II. Create dataset class
class GTPDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# III. Create data loader to generate batches
def create_dataloader_v1(txt, batch_size = 4, 
                         max_length = 256, stride = 128,
                         shuffle = True, drop_last = True,
                         num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GTPDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader

max_length = 4

# Instantiate dataloader
dataloader = create_dataloader_v1(
    raw_text, batch_size = 8, max_length = max_length,
    stride = max_length, shuffle = False
)

# Create iterator object from the dataloader
data_iter = iter(dataloader)

#next() function to access the actual data in data_iter
inputs, targets = next(data_iter)

# III. From token IDs to embedding vectors
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# IV. Create positional embedding layer with same length as token_embedding_layer
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# V. Create input embeddings
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)