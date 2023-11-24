import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy
nlp = spacy.load('en_core_web_sm')
# Download necessary NLTK datasets
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# Define the preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Load the datasets
sarcasm_headlines_v1 = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
sarcasm_headlines_v2 = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)

# Apply the preprocessing function to the headline column of both datasets
sarcasm_headlines_v1['headline_clean'] = sarcasm_headlines_v1['headline'].apply(preprocess_text)
sarcasm_headlines_v2['headline_clean'] = sarcasm_headlines_v2['headline'].apply(preprocess_text)

# Combine the datasets
combined_sarcasm_headlines = pd.concat([sarcasm_headlines_v1, sarcasm_headlines_v2], ignore_index=True)

# Save the combined preprocessed dataset to a new file
combined_sarcasm_headlines.to_csv('combined_preprocessed_headlines.csv', index=False)