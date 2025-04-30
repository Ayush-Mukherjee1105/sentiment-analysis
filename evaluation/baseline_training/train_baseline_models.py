# evaluation/baseline_training/train_baseline_models.py

import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import pandas as pd

from config import MAX_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_LABELS, DATASET_PATH, MODEL_SAVE_DIR, LOG_CSV
from utils import set_seed, SentimentDataset, load_balanced_dataset

# Set random seed and device
set_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Models to train
MODEL_LIST = [
    ("bert-base-uncased", "bert_base.pth"),
    ("roberta-base", "roberta_base.pth"),
    ("distilbert-base-uncased", "distilbert_base.pth"),
    ("albert-base-v2", "albert_base.pth")
]

print("[INFO] Loading dataset...")
X_train, X_val, y_train, y_val = load_balanced_dataset(DATASET_PATH)

# Initialize log
log_data = []

# Train each model
for model_name, save_name in MODEL_LIST:
    print(f"\n🚀 Training {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_data = SentimentDataset(X_train, y_train, tokenizer)
    val_data = SentimentDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=mask)
                pred = torch.argmax(outputs.logits, dim=1)
                preds.extend(pred.cpu().numpy())
                truths.extend(labels.cpu().numpy())

        val_acc = accuracy_score(truths, preds)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, save_name))
            print(f"[BEST MODEL SAVED] for {model_name} with Acc: {val_acc:.4f}")

    log_data.append({
        "Model": model_name,
        "Best_Val_Accuracy": round(best_val_acc, 4)
    })

# Save log
pd.DataFrame(log_data).to_csv(LOG_CSV, index=False)
print("\n✅ All models trained. Results saved to:", LOG_CSV)
