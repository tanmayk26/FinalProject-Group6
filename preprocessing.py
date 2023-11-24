import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
#from tqdm import tqdm

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
