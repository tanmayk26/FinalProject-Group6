import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch import nn
from transformers import RobertaTokenizerFast, RobertaModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import argparse

plt.style.use('ggplot')
BATCH_SIZE = 32
MAX_LEN = 128  
model_save_name = 'trans-roberta-model.pt'

parser = argparse.ArgumentParser()
parser.add_argument('-Train', action='store_true')
args = parser.parse_args()

TRAIN_MODEL = args.Train
def load_and_preprocess(data_paths):
    data_frames = [pd.read_json(path, lines=True) for path in data_paths]
    data = pd.concat(data_frames)

    data['headline'] = data['headline'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', str(x)).lower())
    data['headline'] = data['headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))

    return data

data_paths = ['Sarcasm_Headlines_Dataset.json', 'Sarcasm_Headlines_Dataset_v2.json']
data = load_and_preprocess(data_paths)

