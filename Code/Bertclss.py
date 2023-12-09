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
#--------------MODEL------------------


class BertCNN(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, num_filters=100, filter_sizes=[3, 4, 5]):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, self.bert.config.hidden_size)) for k in filter_sizes]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state.unsqueeze(1)  # Add channel dimension
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

class BertMLP(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, hidden_size=50):
        super(BertMLP, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        x = self.dropout(torch.relu(self.fc1(pooled_output)))
        logits = self.fc2(x)
        return logits

class BertLSTM(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, hidden_size=50, num_layers=1):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)
        pooled_output = torch.mean(lstm_output, 1)
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

class BertRNN(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, hidden_size=50, num_layers=1):
        super(BertRNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.rnn = nn.RNN(self.bert.config.hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        rnn_output, _ = self.rnn(bert_output.last_hidden_state)
        pooled_output = torch.mean(rnn_output, 1)
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

#--------------TRAINNING------------

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy

