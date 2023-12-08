# # Import necessary libraries
# import pandas as pd
# import numpy as np
# import spacy
# from spacy.lang.en.stop_words import STOP_WORDS
# from gensim.models import KeyedVectors
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix
# from textblob import TextBlob
#
# # Load Spacy's English language model
# nlp = spacy.load('en_core_web_sm')
#
# # Preprocess and clean text
# def preprocess_text(text):
#     doc = nlp(text.lower())
#     lemmatized = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
#     return ' '.join(lemmatized)
#
# # Load the sarcasm datasets
# sarcasm_df = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
#
# # Apply the preprocess_text function to clean the headlines
# sarcasm_df['clean_headline'] = sarcasm_df['headline'].apply(preprocess_text)
#
# # Sentiment Analysis
# sarcasm_df['sentiment'] = sarcasm_df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
#
# # Feature Engineering
# sarcasm_df['headline_length'] = sarcasm_df['headline'].apply(len)
# sarcasm_df['word_count'] = sarcasm_df['headline'].apply(lambda x: len(x.split()))
# sarcasm_df['exclamation_mark'] = sarcasm_df['headline'].apply(lambda x: '!' in x)
# sarcasm_df['question_mark'] = sarcasm_df['headline'].apply(lambda x: '?' in x)
#
# # Load pre-trained GloVe word embeddings
# glove_vectors = KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec.txt', binary=False)
#
# # Function to convert headlines into averaged word vector representation
# def headline_to_avg_vector(headline):
#     words = headline.split()
#     word_vectors = [glove_vectors[word] for word in words if word in glove_vectors]
#     if len(word_vectors) == 0:
#         return np.zeros(glove_vectors.vector_size)
#     else:
#         return np.mean(word_vectors, axis=0)
#
# # Apply the function to the cleaned headlines
# sarcasm_df['headline_vector'] = sarcasm_df['clean_headline'].apply(headline_to_avg_vector)
#
# # Split the data into training and testing sets
# X = np.stack(sarcasm_df['headline_vector'].values)
# y = sarcasm_df['is_sarcastic'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define a list of classifiers to experiment with
# classifiers = [
#     LogisticRegression(),
#     MultinomialNB(),
#     MLPClassifier()
# ]
#
# # Evaluate each classifier
# for clf in classifiers:
#     pipeline = make_pipeline(StandardScaler(), clf)
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     print(f'Classifier: {clf.__class__.__name__}')
#     print(classification_report(y_test, y_pred))
#     print(confusion_matrix(y_test, y_pred))
#     print('---')
#
# #further work on advance models(bert)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize, pos_tag, ngrams
from nltk.corpus import stopwords
import re
from collections import Counter
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from tqdm.auto import tqdm
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the datasets
data_v1 = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
data_v2 = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)



# Display the head and description of both datasets
print('First Dataset Head:')
print(data_v1.head())
print('\nFirst Dataset Description:')
print(data_v1.describe(include='all'))

print('\nSecond Dataset Head:')
print(data_v2.head())
print('\nSecond Dataset Description:')
print(data_v2.describe(include='all'))

# Check the balance of sarcastic and non-sarcastic labels in each dataset
balance_v1 = data_v1['is_sarcastic'].value_counts(normalize=True)
balance_v2 = data_v2['is_sarcastic'].value_counts(normalize=True)
print('Balance of labels in first dataset (v1):')
print(balance_v1)
print('\nBalance of labels in second dataset (v2):')
print(balance_v2)

# Check for unique and common headlines
unique_headlines_v1 = data_v1['headline'].nunique()
unique_headlines_v2 = data_v2['headline'].nunique()
common_headlines = pd.Series(list(set(data_v1['headline']).intersection(set(data_v2['headline']))))
print('\nNumber of unique headlines in first dataset (v1):', unique_headlines_v1)
print('Number of unique headlines in second dataset (v2):', unique_headlines_v2)
print('\nNumber of common headlines between the datasets:', common_headlines.size)

# Data Cleaning and Preprocessing
data_v1.dropna(inplace=True)
data_v2.dropna(inplace=True)
common_headlines = set(data_v1['headline'])
data_v2_unique = data_v2[~data_v2['headline'].isin(common_headlines)]

# Merge the datasets
merged_data = pd.concat([data_v1, data_v2_unique]).reset_index(drop=True)

# Check the balance of the merged dataset
balance_merged = merged_data['is_sarcastic'].value_counts(normalize=True)
print('\nNumber of headlines after merging and removing duplicates:', merged_data.shape[0])
print('Balance of labels in the merged dataset:')
print(balance_merged)

# Save the merged dataset
merged_data.to_csv('cleaned_merged_sarcasm_dataset.csv', index=False)
print('\nMerged dataset saved as cleaned_merged_sarcasm_dataset.csv')

# Text Length Analysis
merged_data['headline_length'] = merged_data['headline'].apply(len)
sns.boxplot(x='is_sarcastic', y='headline_length', data=merged_data)
plt.title('Distribution of Headline Lengths by Sarcasm Label')
plt.show()

# Exclamation and Question Marks Analysis
merged_data['exclamation_mark'] = merged_data['headline'].apply(lambda x: x.count('!'))
merged_data['question_mark'] = merged_data['headline'].apply(lambda x: x.count('?'))

sns.countplot(x='exclamation_mark', hue='is_sarcastic', data=merged_data)
plt.title('Distribution of Exclamation Marks in Headlines')
plt.show()

sns.countplot(x='question_mark', hue='is_sarcastic', data=merged_data)
plt.title('Distribution of Question Marks in Headlines')
plt.show()

# Correcting Top Unigrams in Headlines
def plot_top_unigrams(text, title, n=20):
    # Tokenize the text to words
    tokens = [word for headline in text for word in word_tokenize(headline) if word.isalpha()]
    # Calculate frequency distribution
    fdist = nltk.FreqDist(tokens)
    # Prepare for plotting
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})
    d = words_df.nlargest(columns="count", n=n)
    # Plot
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel = 'Count')
    plt.title(title)
    plt.show()

# Now use the function to plot the top words for sarcastic and non-sarcastic headlines
plot_top_unigrams(merged_data[merged_data['is_sarcastic'] == 1]['headline'], 'Top Unigrams in Sarcastic Headlines')
plot_top_unigrams(merged_data[merged_data['is_sarcastic'] == 0]['headline'], 'Top Unigrams in Non-Sarcastic Headlines')

# Define function to calculate sentiment polarity
def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply the sentiment analysis
merged_data['sentiment_polarity'] = merged_data['headline'].apply(calculate_sentiment)

# Visualize the sentiment polarity distribution
sns.boxplot(x='is_sarcastic', y='sentiment_polarity', data=merged_data)
plt.title('Sentiment Polarity Distribution by Sarcasm Label')
plt.show()

# Define function to calculate multiple punctuation count
def count_multiple_punctuation(text):
    return len(re.findall(r'[\?!.]{2,}', text))

# Apply the function to count multiple punctuations
merged_data['multiple_punct_count'] = merged_data['headline'].apply(count_multiple_punctuation)

# Visualize the distribution of multiple consecutive punctuations
sns.countplot(x='multiple_punct_count', hue='is_sarcastic', data=merged_data)
plt.title('Distribution of Multiple Consecutive Punctuation Marks in Headlines')
plt.show()

# Define function to flag large numbers
def contains_large_number(text):
    return int(bool(re.search(r'\b\d{5,}\b', text)))

# Apply the function to flag large numbers
merged_data['large_number_flag'] = merged_data['headline'].apply(contains_large_number)

# Visualize the presence of large numbers in headlines
sns.countplot(x='large_number_flag', hue='is_sarcastic', data=merged_data)
plt.title('Presence of Large Numbers in Headlines')
plt.show()

# Define function to calculate all caps words count
def count_all_caps(text):
    return sum(word.isupper() for word in text.split() if len(word) > 1)

# Apply the function to count all caps words
merged_data['all_caps_count'] = merged_data['headline'].apply(count_all_caps)

# Visualize the use of all caps words
sns.boxplot(x='is_sarcastic', y='all_caps_count', data=merged_data)
plt.title('All Caps Word Count Distribution by Sarcasm Label')
plt.show()

# Define function to calculate punctuation count
def calculate_punctuation_count(text):
    return len(re.findall(r'[^\w\s]', text))

# Apply the function to calculate punctuation count
merged_data['punctuation_count'] = merged_data['headline'].apply(calculate_punctuation_count)

# Visualize the punctuation count distribution
sns.boxplot(x='is_sarcastic', y='punctuation_count', data=merged_data)
plt.title('Punctuation Count Distribution by Sarcasm Label')
plt.show()

# Tokenize and remove stop words from each headline
merged_data['filtered_tokens'] = merged_data['headline'].apply(
    lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]
)

# Top Words Analysis
def plot_top_words(filtered_tokens_series, title):
    all_words = [word for tokens in filtered_tokens_series for word in tokens]
    word_freq = Counter(all_words)
    top_words_df = pd.DataFrame(word_freq.most_common(20), columns=['word', 'count'])
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='word', data=top_words_df)
    plt.title(title)
    plt.show()

# Plot top words for sarcastic and non-sarcastic headlines
plot_top_words(merged_data[merged_data['is_sarcastic'] == 1]['filtered_tokens'], 'Top Words in Sarcastic Headlines')
plot_top_words(merged_data[merged_data['is_sarcastic'] == 0]['filtered_tokens'], 'Top Words in Non-Sarcastic Headlines')

# N-grams Analysis
def plot_top_ngrams(filtered_tokens_series, title, n=2):
    all_ngrams = ngrams([word for tokens in filtered_tokens_series for word in tokens], n)
    ngram_freq = Counter(all_ngrams)
    top_ngrams_df = pd.DataFrame(ngram_freq.most_common(20), columns=['n_gram', 'count'])
    top_ngrams_df['n_gram'] = top_ngrams_df['n_gram'].apply(lambda x: ' '.join(x))
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='n_gram', data=top_ngrams_df)
    plt.title(title)
    plt.show()

# Plot top n-grams for sarcastic and non-sarcastic headlines
plot_top_ngrams(merged_data[merged_data['is_sarcastic'] == 1]['filtered_tokens'], 'Top Bigrams in Sarcastic Headlines', n=2)
plot_top_ngrams(merged_data[merged_data['is_sarcastic'] == 0]['filtered_tokens'], 'Top Bigrams in Non-Sarcastic Headlines', n=2)

# Save the enhanced dataset with new features
merged_data.to_csv('enhanced_sarcasm_dataset.csv', index=False)
print('\nEnhanced dataset saved as enhanced_sarcasm_dataset.csv')

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(merged_data['headline']).toarray()

# Word2Vec Model Training
word2vec_model = Word2Vec(merged_data['filtered_tokens'], vector_size=100, window=5, min_count=2, workers=4)


# Split the data for the machine learning model
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    merged_data['is_sarcastic'],
    test_size=0.2,
    random_state=42
)
##Baseline Models
#1LogisticRegression
from sklearn.linear_model import LogisticRegression
# Initialize the model
lr_model = LogisticRegression(max_iter=1000)
# Train the model
lr_model.fit(X_train, y_train)
# Predict on the test set
y_pred_lr = lr_model.predict(X_test)
# Calculate accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr}")

#2NaiveBayes
from sklearn.naive_bayes import MultinomialNB
# Initialize the model
nb_model = MultinomialNB()
# Train the model
nb_model.fit(X_train, y_train)
# Predict on the test set
y_pred_nb = nb_model.predict(X_test)
# Calculate accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")

#SVM
from sklearn.svm import SVC
# Initialize the model
svm_model = SVC()
# Train the model
svm_model.fit(X_train, y_train)
# Predict on the test set
y_pred_svm = svm_model.predict(X_test)
# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm}")

##Ensemble Models
#1RandomForest
from sklearn.ensemble import RandomForestClassifier
# Initialize the model
rf_model = RandomForestClassifier()
# Train the model
rf_model.fit(X_train, y_train)
# Predict on the test set
y_pred_rf = rf_model.predict(X_test)
# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")

#2GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
# Initialize the model
gb_model = GradientBoostingClassifier()
# Train the model
gb_model.fit(X_train, y_train)
# Predict on the test set
y_pred_gb = gb_model.predict(X_test)
# Calculate accuracy
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb}")

##Deep Learning Approaches
#1LSTM (Long Short-Term Memory networks):
