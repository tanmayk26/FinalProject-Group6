import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
#from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import numpy as np
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load Spacy's English language model
nlp = spacy.load('en_core_web_sm')

# Load the sarcasm datasets from the provided JSON files
sarcasm_df = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
sarcasm_df_v2 = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)

# Show the head of the datasets to inspect the data
print(sarcasm_df.head())
print(sarcasm_df_v2.head())

# Check for missing values
print(sarcasm_df.isnull().sum())
print(sarcasm_df_v2.isnull().sum())

# Check for duplicates
print(sarcasm_df.duplicated().sum())
print(sarcasm_df_v2.duplicated().sum())

# Remove duplicates
sarcasm_df = sarcasm_df.drop_duplicates()
sarcasm_df_v2 = sarcasm_df_v2.drop_duplicates()

# Show the number of rows remaining after duplicate removal
print('Number of rows in sarcasm_df:', sarcasm_df.shape[0])
print('Number of rows in sarcasm_df_v2:', sarcasm_df_v2.shape[0])

# Define a text preprocessing function
def preprocess_text(text):
    # Process the text using Spacy
    doc = nlp(text.lower())
    # Lemmatize and retain words (excluding spaces)
    lemmatized = [token.lemma_ for token in doc if not token.is_space]
    return ' '.join(lemmatized)

# Apply the preprocessing function to the headline column
sarcasm_df['clean_headline'] = sarcasm_df['headline'].apply(preprocess_text)
sarcasm_df_v2['clean_headline'] = sarcasm_df_v2['headline'].apply(preprocess_text)

# Extract additional features
sarcasm_df['headline_length'] = sarcasm_df['headline'].apply(len)
sarcasm_df['word_count'] = sarcasm_df['headline'].apply(lambda x: len(x.split()))
sarcasm_df_v2['headline_length'] = sarcasm_df_v2['headline'].apply(len)
sarcasm_df_v2['word_count'] = sarcasm_df_v2['headline'].apply(lambda x: len(x.split()))

# Show the head of the updated datasets
print(sarcasm_df.head())
print(sarcasm_df_v2.head())

# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Plot the distribution of sarcastic vs non-sarcastic headlines
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
sns.countplot(x='is_sarcastic', data=sarcasm_df)
plt.title('Distribution of Sarcastic vs Non-Sarcastic Headlines (Dataset 1)')

plt.subplot(1, 2, 2)
sns.countplot(x='is_sarcastic', data=sarcasm_df_v2)
plt.title('Distribution of Sarcastic vs Non-Sarcastic Headlines (Dataset 2)')

plt.tight_layout()
plt.show()

#EDA
# Function to plot most common words
def plot_common_words(text, num_words=20):
    word_freq = Counter(" ".join(text).split()).most_common(num_words)
    words, counts = zip(*word_freq)
    plt.figure(figsize=(12, 8))
    plt.bar(words, counts)
    plt.xticks(rotation=45)
    plt.show()

# Plot common words in sarcastic headlines
plot_common_words(sarcasm_df[sarcasm_df['is_sarcastic'] == 1]['clean_headline'])

# Plot common words in non-sarcastic headlines
plot_common_words(sarcasm_df[sarcasm_df['is_sarcastic'] == 0]['clean_headline'])

# Headline Length Analysis
plt.figure(figsize=(12, 6))
sns.histplot(sarcasm_df['headline_length'], bins=30, kde=True, color='blue', label='Sarcastic')
sns.histplot(sarcasm_df_v2['headline_length'], bins=30, kde=True, color='red', label='Non-Sarcastic')
plt.title('Headline Length Distribution')
plt.legend()
plt.show()

#Feature Engineering
# TF-IDF Vectorization
#tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
#X_tfidf = tfidf_vectorizer.fit_transform(sarcasm_df['clean_headline'])
#y = sarcasm_df['is_sarcastic']

# TF-IDF with n-grams
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = tfidf_vectorizer.fit_transform(sarcasm_df['clean_headline'])
y = sarcasm_df['is_sarcastic']


#Sentiment score
def get_sentiment(text):
    # Get the sentiment polarity
    return TextBlob(text).sentiment.polarity

# Apply the function to get sentiment scores
sarcasm_df['sentiment'] = sarcasm_df['clean_headline'].apply(get_sentiment)

#Basic Model Building
# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr))

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes:")
print(classification_report(y_test, y_pred_nb))