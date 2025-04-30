# training/train_custom_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# Dataset class
class SocialDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        print("[INFO] Tokenizer loaded ✅")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=128
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, self.labels[idx]

# Upgraded Model class
class LightBiLSTM_BERTMix(nn.Module):
    def __init__(self):
        super(LightBiLSTM_BERTMix, self).__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.lstm = nn.LSTM(768, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.hidden = nn.Linear(256, 128)
        self.fc = nn.Linear(128, 2)  # 2 classes: negative, positive

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(bert_out)
        pooled = torch.max(lstm_out, 1)[0]
        dropped = self.dropout(pooled)
        hidden_out = torch.relu(self.hidden(dropped))
        output = self.fc(hidden_out)
        return output

# Load and balance dataset
print("[INFO] Loading dataset...")

df = pd.read_csv('datasets/training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

positive_df = df[df['target'] == 4]
negative_df = df[df['target'] == 0]

positive_sample = positive_df.sample(100000, random_state=42)
negative_sample = negative_df.sample(100000, random_state=42)

balanced_df = pd.concat([positive_sample, negative_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"[INFO] Balanced large dataset size: {len(balanced_df)} posts (100k positive + 100k negative)")

label_map = {0: 0, 4: 1}
texts = balanced_df['text'].tolist()
labels = balanced_df['target'].map(label_map).tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

train_dataset = SocialDataset(train_texts, train_labels)
val_dataset = SocialDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize model
model = LightBiLSTM_BERTMix().to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0

# Training loop
print("[INFO] Starting training... 🚀")

for epoch in range(10):  # 10 epochs
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)

    for input_ids, attention_mask, labels in progress_bar:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    # Validation phase
    model.eval()
    val_preds = []
    val_labels_list = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels_list, val_preds)
    val_f1 = f1_score(val_labels_list, val_preds, average='weighted')

    print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(model.state_dict(), 'model/optimized_model.pth')
        print(f"[BEST MODEL SAVED] at Epoch {epoch+1} with Val Accuracy: {val_acc:.4f}")

print("[INFO] Training finished ✅")
print(f"[INFO] Best Validation Accuracy Achieved: {best_val_acc:.4f}")
