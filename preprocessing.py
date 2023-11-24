import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm.auto import tqdm

import nltk

# Load the dataset files
sarcasm_headlines_v1 = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
sarcasm_headlines_v2 = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)

print(sarcasm_headlines_v1.head())
print(sarcasm_headlines_v2.head())

#Visualizing the data
sarcasm_headlines_v1 = sarcasm_headlines_v1[['headline', 'is_sarcastic']]
sarcasm_headlines_v2 = sarcasm_headlines_v2[['headline', 'is_sarcastic']]
print(sarcasm_headlines_v1.head())
print(sarcasm_headlines_v2.head())

#visualizing the distribution of sarcastic and non-sarcastic headlines
print(sarcasm_headlines_v1['is_sarcastic'].value_counts())
print(sarcasm_headlines_v2['is_sarcastic'].value_counts())

# Combining the datasets for visualization
combined_data = pd.concat([sarcasm_headlines_v1, sarcasm_headlines_v2])

# Plot the distribution of sarcastic vs non-sarcastic headlines
plt.figure(figsize=(8, 5))
sns.countplot(x='is_sarcastic', data=combined_data)
plt.title('Distribution of Sarcastic vs Non-Sarcastic Headlines')
plt.xlabel('Is Sarcastic')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Sarcastic', 'Sarcastic'])
plt.show()


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tqdm.pandas()
# Function to clean the headlines
def clean_headline(headline):
    # Remove special characters and numbers
    headline = re.sub(r'[^A-Za-z ]+', '', headline)
    # Tokenize
    words = word_tokenize(headline)
    # Lowercase and remove stop words
    words = [word.lower() for word in words if word.lower() not in stop_words]
    # Stemming
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


# Apply the cleaning function to the headlines
sarcasm_headlines_v1['headline_clean'] = sarcasm_headlines_v1['headline'].progress_apply(clean_headline)
sarcasm_headlines_v2['headline_clean'] = sarcasm_headlines_v2['headline'].progress_apply(clean_headline)

# Display the cleaned headlines
print(sarcasm_headlines_v1[['headline', 'headline_clean']].head())
print(sarcasm_headlines_v2[['headline', 'headline_clean']].head())