import os
import gc
import json
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate

# Initialize evaluation metrics
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
random.seed(42)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def set_random_seed(seed: int):
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

def load_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data).dropna().to_dict(orient='records')

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, source_column, target_column, max_length=480):
        self.data = data
        self.tokenizer = tokenizer
        self.source_column = source_column
        self.target_column = target_column
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item[self.source_column]
        target_text = item[self.target_column]

        input_tokens = self.tokenizer.encode(
            input_text, max_length=self.max_length, truncation=True, padding='max_length')
        target_tokens = self.tokenizer.encode(
            target_text, max_length=self.max_length, truncation=True, padding='max_length')

        return torch.tensor(input_tokens), torch.tensor(target_tokens)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded

def prepare_data(file_path, source_column, target_column):
    dataset = pd.DataFrame(json.load(open(file_path))).dropna()
    train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.7, random_state=42)

    for split, name in zip([train_data, valid_data, test_data], ["train", "val", "test"]):
        split.to_json(f"{name}.json", orient='records')

    return [load_dataset(f"{name}.json") for name in ["train", "val", "test"]]


def train_model(model, tokenizer, train_loader, valid_loader, device, epochs, lr, patience, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_valid_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Train Loss: {total_loss / len(train_loader):.4f}")

        valid_loss = validate_model(model, valid_loader, device)
        print(f"Validation Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

def validate_model(model, valid_loader, device):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(input_ids=inputs, labels=targets)
            valid_loss += outputs.loss.item()
    return valid_loss / len(valid_loader)

def evaluate_model(model, test_loader, tokenizer, device, source_column, target_column):
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model.generate(input_ids=inputs, max_length=50)
            predictions.extend(outputs)
            references.extend(targets)

    predictions = [tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
    references = [tokenizer.decode(r, skip_special_tokens=True) for r in references]

    return predictions, references

def calculate_metrics(predictions, references):
    rouge = Rouge()
    scores = {
        "BLEU": sentence_bleu([r.split()], p.split(), weights=(0.25, 0.25, 0.25, 0.25)) for p, r in zip(predictions, references)
    }

    rouge_scores = rouge.get_scores(predictions, references, avg=True)
    scores.update({
        "ROUGE-1": rouge_scores['rouge-1']['f'],
        "ROUGE-2": rouge_scores['rouge-2']['f'],
        "ROUGE-L": rouge_scores['rouge-l']['f']
    })
    return scores

if __name__ == "__main__":
    data_path = "data/version_3.json"
    source_column = "concated_transcript"
    target_column = "dcs"

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data, valid_data, test_data = prepare_data(data_path, source_column, target_column)

    train_dataset = CustomDataset(train_data, tokenizer, source_column, target_column)
    valid_dataset = CustomDataset(valid_data, tokenizer, source_column, target_column)
    test_dataset = CustomDataset(test_data, tokenizer, source_column, target_column)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    train_model(model, tokenizer, train_loader, valid_loader, device, epochs=10, lr=5e-5, patience=3, save_path="best_model.pth")

    model.load_state_dict(torch.load("best_model.pth"))
    predictions, references = evaluate_model(model, test_loader, tokenizer, device, source_column, target_column)
    metrics = calculate_metrics(predictions, references)

    print("Evaluation Metrics:", metrics)
