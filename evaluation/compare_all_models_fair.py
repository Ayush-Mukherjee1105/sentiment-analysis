import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'baseline_training')))
from config import BATCH_SIZE, NUM_LABELS, DATASET_PATH, MODEL_SAVE_DIR

from utils import set_seed, SentimentDataset, load_balanced_dataset

# Load test data
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, X_val, y_train, y_val = load_balanced_dataset(DATASET_PATH)
tokenizers = {}

# Model registry
MODEL_PATHS = {
    "YourModel": "custom_model.pth",
    "BERT-base": "bert_base.pth",
    "RoBERTa-base": "roberta_base.pth",
    "DistilBERT-base": "distilbert_base.pth",
    "ALBERT-base-v2": "albert_base.pth"
}

MODEL_CLASSES = {
    "YourModel": "distilbert-base-uncased",
    "BERT-base": "bert-base-uncased",
    "RoBERTa-base": "roberta-base",
    "DistilBERT-base": "distilbert-base-uncased",
    "ALBERT-base-v2": "albert-base-v2"
}

# Custom model
class LightBiLSTM_BERTMix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.lstm = torch.nn.LSTM(input_size=768, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(128 * 2, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        out = lstm_out[:, -1, :]
        return self.fc(out)

# Evaluate function
def evaluate_model(name, model, tokenizer):
    dataset = SentimentDataset(X_val, y_val, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    model.to(device)
    model.eval()

    preds, truths = [], []
    times = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            if name == "YourModel":
                output = model(input_ids, attention_mask)
            else:
                output = model(input_ids=input_ids, attention_mask=attention_mask).logits

            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)

            pred = torch.argmax(output, dim=1)
            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())
            times.append(elapsed)

    report = classification_report(truths, preds, target_names=['Negative', 'Positive'], output_dict=True)
    return {
        'Model': name,
        'Accuracy': round(report['accuracy'], 4),
        'F1-macro': round(f1_score(truths, preds, average='macro'), 4),
        'Precision': round(report['weighted avg']['precision'], 4),
        'Recall': round(report['weighted avg']['recall'], 4),
        'Latency(ms/sample)': round(np.mean(times) / BATCH_SIZE, 2)
    }

# Run evaluation
results = []
for name, file in MODEL_PATHS.items():
    print(f"🔍 Evaluating {name}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CLASSES[name])
    tokenizers[name] = tokenizer

    if name == "YourModel":
        model = LightBiLSTM_BERTMix()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_CLASSES[name], num_labels=NUM_LABELS)

    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, file), map_location=device))
    results.append(evaluate_model(name, tokenizer=tokenizers[name], model=model))

# Save + Plot
df = pd.DataFrame(results)
df.to_csv("evaluation/baseline_training/final_comparison_results.csv", index=False)
print("✅ Saved comparison to final_comparison_results.csv")

# Plot metrics
sns.set(style='whitegrid')
melted = df.melt(id_vars='Model', value_vars=['Accuracy', 'F1-macro', 'Precision', 'Recall'])
plt.figure(figsize=(10, 6))
sns.barplot(data=melted, x='variable', y='value', hue='Model')
plt.title("Fair Model Comparison Metrics")
plt.ylabel("Score")
plt.savefig("evaluation/baseline_training/fair_metrics_plot.png")
plt.close()

# Plot latency
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x='Model', y='Latency(ms/sample)', palette='mako')
plt.title("Inference Latency (Lower is Better)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("evaluation/baseline_training/fair_latency_plot.png")
plt.close()

print("📊 Plots saved: fair_metrics_plot.png & fair_latency_plot.png")
