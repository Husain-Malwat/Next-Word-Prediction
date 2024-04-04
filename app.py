import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import zipfile
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


# Define the model
class NextChar(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.size(0), -1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x
    
def generate_text(model, itos, stoi, block_size, max_len=10, context_sentence=None):
    if context_sentence is None:
        context = torch.zeros(1, block_size, dtype=torch.long).to(device)
    else:
        context = [0] * (block_size - len(context_sentence)) + [stoi[word] for word in context_sentence]
        context = torch.tensor([context], dtype=torch.long).to(device)

    text = ''
    for i in range(max_len):
        y_pred = model(context)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        if(ix==8):
            ch = 'e'
        else:
            ch = itos[ix]
            
        if ch == '.':
            break
        text += ch
        context = torch.cat([context[:, 1:], torch.tensor([[ix]], dtype=torch.long).to(device)], dim=1)
    return text

# Function for next k characters prediction
def predict_next_chars(input_text, k, block_size, embedding_size, hidden_size, vocab_size, itos, stoi):
    # Load your MLP model
    block_size = block_size
    emb_dim = embedding_size
    hidden_size = hidden_size
    model = NextChar(block_size, vocab_size, emb_dim, hidden_size)
    filename = r"C:\Users\vedan\OneDrive\Desktop\ML\ML NipunBatra\Assignment 3\ModelsForAss3"+ f"\Model-b{block_size}-e{emb_dim}-h{hidden_size}.pth"
    print(filename)
    try:
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        if torch.cuda.is_available():
            model = model.to(torch.device('cuda'))
    except FileNotFoundError:
        print("The specified file does not exist.")
    except Exception as e:
        print(f"An exception occurred: {e}")
        print("Failed to load the model.")
    if len(input_text) > block_size:
        print("Contest size id",block_size)
        return "Please put the input text size smaller than the block size!"
    
    generated_text = generate_text(model, itos, stoi, block_size, k, input_text.lower())
    return input_text + generated_text

# Streamlit UI
st.title("Next Character Predictor")

input_text = st.text_input("Enter seed text:", "")
k = st.slider("Select number of characters to predict:", 1, 100, 5)
block_size = st.slider("Select block size:", 1, 20, 3)
embedding_size = st.slider("Select embedding size:", 1, 64, 50)
hidden_size = st.slider("Select hidden layer size:", 1, 64, 50)

zip_file = r'archive.zip'
extract_to = './'

extract_zip(zip_file, extract_to)
filename = "./paul_graham_essays.txt"


text = ""
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()
words = text.split()
words = " ".join(words[:1000])
words = words.lower()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)
# Move data to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Define model hyperparameters
vocab_size = len(stoi) + 1  # Add 1 for padding

if st.button("Predict"):
    if input_text:
        predicted_chars = predict_next_chars(input_text, k, block_size, embedding_size,hidden_size, vocab_size,itos,stoi)
        st.write(f"Predicted next {k} characters:", predicted_chars)
    else:
        st.warning("Please enter some text.")
