import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch import nn
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import argparse
from transformers import DataCollatorWithPadding
import torch
import string
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources
nltk.download('omw-1.4')
nltk.download('wordnet')

class Config:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.general = {
            'max_length': 128,
            'target_columns': ['is_sarcastic'],
            'train_batch_size': 32,
            'valid_batch_size': 16,
            'n_workers': 2
        }

cfg = Config()

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

class SarcasmDataset(Dataset):
    def __init__(self, cfg, df, train=True):
        self.cfg = cfg
        self.df = df
        self.df['headline'] = self.df['headline'].apply(preprocess_text)
        self.texts = self.df['headline'].values
        self.labels = None
        if train and cfg.general['target_columns'][0] in df.columns:
            self.labels = df[cfg.general['target_columns']].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        inputs = self.cfg.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.cfg.general.max_length,
            pad_to_max_length=True,
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[item], dtype=torch.float)
            return inputs, label
        return inputs

def get_train_dataloader(cfg, df):
    dataset = SarcasmDataset(cfg, df)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.general['train_batch_size'],
        num_workers=cfg.general['n_workers'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


df1 = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
df2 = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)

df = pd.concat([df1, df2], ignore_index=True)
train_dataloader = get_train_dataloader(cfg, df)
print(df.head())
