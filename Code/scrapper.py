import pandas as pd
import os
import bs4
from tqdm import tqdm
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from socket import gaierror
from utils import avail_data

code_dir = os.getcwd()
data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
avail_data(data_dir)
final_data = pd.read_json(os.path.join(data_dir, r'Combined_Headlines.json'))
sarcastic_data = final_data.iloc[final_data.filter(final_data.is_sarcastic == 1).index]
