import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import torch.nn as nn
from transformers import BertModel
from Bertclss import SarcasmDataset,DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

class SarcasmDataset(Dataset):
    def __init__(self, json_file, tokenizer_name='bert-base-uncased', max_length=128):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        with open(json_file, 'r') as file:
            for line in file:
                entry = json.loads(line)
                self.data.append((entry['headline'], entry['is_sarcastic']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded_text['input_ids'].flatten(),
            'attention_mask': encoded_text['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
