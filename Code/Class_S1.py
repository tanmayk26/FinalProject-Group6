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
