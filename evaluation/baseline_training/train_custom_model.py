# evaluation/baseline_training/train_custom_model.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
import pandas as pd
from config import MAX_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_LABELS, SEED, DATASET_PATH, MODEL_SAVE_DIR
from utils import set_seed, SentimentDataset, load_balanced_dataset

# ✅ Model Definition (same as original)
class LightBiLSTM_BERTMix(nn.Module):
    def __init__(self):
        super(LightBiLSTM_BERTMix, self).__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        out = lstm_out[:, -1, :]
        return self.fc(out)

# 🏁 Setup
set_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print("[INFO] Loading dataset...")
X_train, X_val, y_train, y_val = load_balanced_dataset(DATASET_PATH)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
train_data = SentimentDataset(X_train, y_train, tokenizer)
val_data = SentimentDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# Model + optimizer
model = LightBiLSTM_BERTMix().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_acc = 0.0
print("[INFO] Starting training... 🚀")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, mask)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss:.4f} - Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'custom_model.pth'))
        print(f"[BEST MODEL SAVED] at Epoch {epoch + 1} with Val Accuracy: {val_acc:.4f}")

print("✅ Training complete.")
