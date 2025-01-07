import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import warnings
import json
import random
import pickle
import gc
import os
from nltk.translate.meteor_score import meteor_score
from evaluate import load
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd

# Load evaluation metrics
bertscore = load("bertscore")
meteor = evaluate.load('meteor')

# Configure CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set random seed for reproducibility
def set_random_seed(seed: int):
    print(f"Seed: {seed}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(42)

# Load dataset from JSON
def load_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data).reset_index(drop=True)
    return df.to_dict(orient='records')

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", model_max_length=480)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.max_length = 480

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item[source_column]
        target_text = item[target_column]
        input_tokens = self.tokenizer.encode(input_text, truncation=True, max_length=self.max_length, padding="max_length")
        target_tokens = self.tokenizer.encode(target_text, truncation=True, max_length=self.max_length, padding="max_length")
        return torch.tensor(input_tokens), torch.tensor(target_tokens)

# Collate function for DataLoader
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded

# Training configuration
batch_size = 8
num_epochs = 60
learning_rate = 5e-5
source_column = "concated_transcript"
target_column = "mmcs"

# Read and preprocess data
def read_json_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

dataset_path = 'data/version_3.json'
dataset = pd.DataFrame(read_json_data(dataset_path))
train_data, test_data = train_test_split(dataset, test_size=0.2)
valid_data, test_data = train_test_split(test_data, test_size=0.7)

for name, data in [("train", train_data), ("val", valid_data), ("test", test_data)]:
    data.to_json(f"{name}.json")

# Load datasets
train_dataset = CustomDataset(load_dataset("train.json"))
valid_dataset = CustomDataset(load_dataset("val.json"))
test_dataset = CustomDataset(load_dataset("test.json"))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_fn)

print("Data loaded")

# Load GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(train_dataset.tokenizer))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
print("Model loaded")

# Training loop
best_valid_loss = float('inf')
patience, counter = 3, 0
best_model_path = "best_model.pth"

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, targets = (b.to(device) for b in batch)
        optimizer.zero_grad()
        loss = model(inputs, labels=targets).loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss}")

    # Validation
    model.eval()
    valid_loss = sum(model(inputs.to(device), labels=targets.to(device)).loss.item() for inputs, targets in valid_loader) / len(valid_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {valid_loss}")

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# Evaluation
model.load_state_dict(torch.load(best_model_path))
model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        predictions.extend(model.generate(inputs.to(device), max_length=50))

# Decode predictions and compute metrics
decoded_preds = [train_dataset.tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
results = pd.DataFrame({
    "Input Text": [d[source_column] for d in test_data],
    "Actual": [d[target_column] for d in test_data],
    "Generated": decoded_preds
})

# Save results and scores
results.to_csv("predictions.csv", index=False)
print("Predictions saved")
