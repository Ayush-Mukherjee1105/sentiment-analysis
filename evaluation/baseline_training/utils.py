# evaluation/baseline_training/utils.py

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import random
import numpy as np
from config import SEED, MAX_LEN

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx])
        }

def load_balanced_dataset(csv_path, sample_size=50000):
    df = pd.read_csv(csv_path, encoding='latin-1', header=None)
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = df[df['target'].isin([0, 4])]
    df['label'] = df['target'].map({0: 0, 4: 1})

    df_neg = df[df['label'] == 0].sample(sample_size // 2, random_state=SEED)
    df_pos = df[df['label'] == 1].sample(sample_size // 2, random_state=SEED)
    df_sample = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    return train_test_split(df_sample['text'].tolist(), df_sample['label'].tolist(), test_size=0.1, random_state=SEED)
