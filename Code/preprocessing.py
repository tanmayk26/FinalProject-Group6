# Import necessary libraries
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from textblob import TextBlob

# Load Spacy's English language model
nlp = spacy.load('en_core_web_sm')

# Preprocess and clean text
def preprocess_text(text):
    doc = nlp(text.lower())
    lemmatized = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
    return ' '.join(lemmatized)

# Load the sarcasm datasets
sarcasm_df = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)

# Apply the preprocess_text function to clean the headlines
sarcasm_df['clean_headline'] = sarcasm_df['headline'].apply(preprocess_text)

# Sentiment Analysis
sarcasm_df['sentiment'] = sarcasm_df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Feature Engineering
sarcasm_df['headline_length'] = sarcasm_df['headline'].apply(len)
sarcasm_df['word_count'] = sarcasm_df['headline'].apply(lambda x: len(x.split()))
sarcasm_df['exclamation_mark'] = sarcasm_df['headline'].apply(lambda x: '!' in x)
sarcasm_df['question_mark'] = sarcasm_df['headline'].apply(lambda x: '?' in x)

# Load pre-trained GloVe word embeddings
glove_vectors = KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec.txt', binary=False)

# Function to convert headlines into averaged word vector representation
def headline_to_avg_vector(headline):
    words = headline.split()
    word_vectors = [glove_vectors[word] for word in words if word in glove_vectors]
    if len(word_vectors) == 0:
        return np.zeros(glove_vectors.vector_size)
    else:
        return np.mean(word_vectors, axis=0)

# Apply the function to the cleaned headlines
sarcasm_df['headline_vector'] = sarcasm_df['clean_headline'].apply(headline_to_avg_vector)

# Split the data into training and testing sets
X = np.stack(sarcasm_df['headline_vector'].values)
y = sarcasm_df['is_sarcastic'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of classifiers to experiment with
classifiers = [
    LogisticRegression(),
    MultinomialNB(),
    MLPClassifier()
]

# Evaluate each classifier
for clf in classifiers:
    pipeline = make_pipeline(StandardScaler(), clf)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f'Classifier: {clf.__class__.__name__}')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('---')

#further work on advance models(bert)    