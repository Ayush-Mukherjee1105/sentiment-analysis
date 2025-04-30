# model/infer.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model class
class LightBiLSTM_BERTMix(nn.Module):
    def __init__(self):
        super(LightBiLSTM_BERTMix, self).__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.lstm = nn.LSTM(768, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.hidden = nn.Linear(256, 128)
        self.fc = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(bert_out)
        pooled = torch.max(lstm_out, 1)[0]
        dropped = self.dropout(pooled)
        hidden_out = torch.relu(self.hidden(dropped))
        output = self.fc(hidden_out)
        return output

# Initialize model
model = LightBiLSTM_BERTMix().to(device)
model.load_state_dict(torch.load('model/optimized_model.pth', map_location=device))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def predict_sentiment(text):
    """
    Predict sentiment for a given text: returns 'positive' or 'negative'
    """
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=128
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = torch.argmax(outputs, dim=1).item()

    if prediction == 1:
        return 'positive'
    else:
        return 'negative'
