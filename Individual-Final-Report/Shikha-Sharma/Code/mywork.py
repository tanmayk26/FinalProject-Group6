import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

def process_text(data):
    lemma = nltk.WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    processed_data = []
    for sent in data:
        words = [lemma.lemmatize(word) for word in sent.split() if word not in stopwords_set]
        processed_data.append(' '.join(words))
    return processed_data
# Function to evaluate the model
def evaluate_model(clf, X_train, y_train, X_test, y_test, name):
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print(f"{name} Train Accuracy: {accuracy_score(y_train, y_train_pred)}")
    print(f"{name} Train F1-score: {f1_score(y_train, y_train_pred)}")
    print(f"{name} Test Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(f"{name} Test F1-score: {f1_score(y_test, y_test_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_test_pred)}")
    filename = f'finalized_{name}_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

# Paths to the dataset files
train_file_path = '../../../Code/Sarcasm_Headlines_Dataset.json'
test_file_path = '../../../Code/Sarcasm_Headlines_Dataset_v2.json'
df_train = pd.read_json(train_file_path, lines=True)
df_test = pd.read_json(test_file_path, lines=True)
df = pd.concat([df_train, df_test], axis=0)

# Split data
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['is_sarcastic'], shuffle=True)
X_train, y_train = process_text(df_train['headline']), df_train['is_sarcastic']
X_test, y_test = process_text(df_test['headline']), df_test['is_sarcastic']

# Vectorization
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# Train Logistic Regression model
clf_lr = LogisticRegression(n_jobs=1, C=1e5)
clf_lr.fit(X_train, y_train)
evaluate_model(clf_lr, X_train, y_train, X_test, y_test, 'LR')

# Train Naive Bayes model
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)
evaluate_model(clf_nb, X_train, y_train, X_test, y_test, 'NB')

num_examples = 3

for _ in range(num_examples):
    idx = np.random.randint(0, len(df_test))
    headline = df_test.iloc[idx]["headline"]
    c_lr = make_pipeline(tfidf, clf_lr)
    class_names = ["Non Sarcastic", "Sarcastic"]
    explainer_lr = LimeTextExplainer(class_names=class_names)
    exp_lr = explainer_lr.explain_instance(headline, c_lr.predict_proba, num_features=10)
    print(f"Logistic Regression - Example {idx}")
    print("Headline:", headline)
    print("Probability (Non sarcastic) =", c_lr.predict_proba([headline])[0, 1])
    print("Probability (sarcastic) =", c_lr.predict_proba([headline])[0, 0])
    print("True Class:", class_names[df_test.iloc[idx]["is_sarcastic"]])
    exp_lr.as_pyplot_figure()
    plt.title(f"LIME Explanation for Example {idx} - Logistic Regression")
    plt.show()

    c_nb = make_pipeline(tfidf, clf_nb)
    explainer_nb = LimeTextExplainer(class_names=class_names)
    exp_nb = explainer_nb.explain_instance(headline, c_nb.predict_proba, num_features=10)
    print(f"Naive Bayes - Example {idx}")
    print("Headline:", headline)
    print("Probability (Non sarcastic) =", c_nb.predict_proba([headline])[0, 1])
    print("Probability (sarcastic) =", c_nb.predict_proba([headline])[0, 0])
    print("True Class:", class_names[df_test.iloc[idx]["is_sarcastic"]])

    exp_nb.as_pyplot_figure()
    plt.title(f"LIME Explanation for Example {idx} - Naive Bayes")
    plt.show()

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import BertModel
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
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = SarcasmDataset('../../../Code/Sarcasm_Headlines_Dataset.json')
    val_dataset = SarcasmDataset('../../../Code/Sarcasm_Headlines_Dataset.json')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = BertRNN()  #  BertMLP, BertLSTM, BertRNN
    model.to(device)
    epochs = 3
    best_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * epochs)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_accuracy:.3f}")
        print(f"Val Loss: {val_loss:.3f}, Val Acc: {val_accuracy:.3f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pt')

    print("Training complete!")
if __name__ == "__main__":
    main()

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizerFast, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re
import nltk
from tqdm import tqdm
nltk.download('stopwords')
nltk.download('wordnet')

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file, tokenizer_name='roberta-base', max_length=None):
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        data = []
        for line in lines:
            item = json.loads(line)
            data.append({
                'text': item['headline'],
                'label': item['is_sarcastic']
            })

        final_df = pd.DataFrame(data)
        final_df = self.preprocess(final_df)
        self.data = final_df
    def preprocess(self, df):
        # Remove numbers
        df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))
        df['text'] = df['text'].str.lower()
        stop_words = set(stopwords.words('english'))
        df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        lemmatizer = WordNetLemmatizer()
        df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

        return df
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data.iloc[idx]
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
class RobertaLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, num_classes=2, freeze_bert=True, dropout_rate=0.5):
        super(RobertaLSTM, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(self.roberta.config.hidden_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(roberta_output.last_hidden_state)
        pooled_output = torch.mean(lstm_output, 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=3,
          early_stopping_patience=5):
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
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
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_correct / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_loss:.3f}, Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.3f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model_roberta.pt')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            print("Early stopping triggered")
            break
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
    avg_accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, avg_accuracy
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = SarcasmDataset('../../../Code/Sarcasm_Headlines_Dataset.json')
    val_dataset = SarcasmDataset('../../../Code/Sarcasm_Headlines_Dataset_v2.json')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = RobertaLSTM().to(device)  # Change to RobertaRNN or RobertaMLP as needed
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=3,
          early_stopping_patience=5)
    print("Training complete!")
if __name__ == "__main__":
    main()
#-----------APP

import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import lime
from lime.lime_text import LimeTextExplainer
import nltk
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from transformers import pipeline

# Ensure the necessary NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')


# Function to process text data
def process_text(data):
    lemma = nltk.WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    processed_data = []
    for sent in data:
        words = [lemma.lemmatize(word) for word in sent.split() if word.lower() not in stopwords_set]
        processed_data.append(' '.join(words))
    return processed_data


# Function to evaluate the model and return metrics
def evaluate_model(clf, X_train, y_train, X_test, y_test):
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)
    return train_accuracy, train_f1, test_accuracy, test_f1, report


# Function for text summarization
def text_summarization(text):
    saved_model = "t5-sarcastic-headline-generator"
    summarizer = pipeline("summarization", model=saved_model)
    summarized_text = summarizer(text, min_length=10, max_length=30)[0]['summary_text']
    return summarized_text


# Function for sarcasm prediction
def predict_sarcasm(text, clf, tfidf):
    processed_text = process_text([text])
    tfidf_text = tfidf.transform(processed_text)
    prediction = clf.predict(tfidf_text)
    return "Sarcastic" if prediction[0] == 1 else "Non Sarcastic"


# Main function for the Streamlit app
def main():
    st.title("Sarcasm Detection in Headlines")

    # Project Overview
    st.header("Project Overview")
    st.write(
        "This project aims to detect sarcasm in news headlines using machine learning. It's an exploration into natural language processing and sentiment analysis.")

    # Sidebar with Project Info
    st.sidebar.header("Project Information")
    st.sidebar.markdown("""
        - **Project Title:** Sarcasm Detection Project
        - **Presented By Group 6:**  
            Shikha Sharma<br>
            Purvi Jain<br>
            Tanmay Kshirsagar
        - **Instructor:** Amir Jafari
        - **Completion Time:** 1 months
    """, unsafe_allow_html=True)

    # Dataset Display
    st.header("Dataset Exploration")
    uploaded_file = st.file_uploader("Upload your dataset (JSON format)", type=["json"])
    if uploaded_file is not None:
        df = pd.read_json(uploaded_file, lines=True)
        if st.checkbox("Show dataset head"):
            st.dataframe(df.head())

        # Data Analysis
        st.subheader("Data Analysis and Visualization")
        if st.checkbox("Show data analysis"):
            st.subheader("Sarcasm Count in Dataset")
            sarcasm_count = df['is_sarcastic'].value_counts()
            st.bar_chart(sarcasm_count)
            st.subheader("Top Words in Sarcastic vs Non-Sarcastic Headlines")
            st.image("img.png", caption="Top Words in Sarcastic Headlines")
            st.image("img_1.png", caption="Top Words in Non-Sarcastic Headlines")

            # Average Word Length Visualization
            st.subheader("Average Word Length in Sarcastic vs Non-Sarcastic Text")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            word = df[df['is_sarcastic'] == 1]['headline'].str.split().apply(lambda x: [len(i) for i in x])
            sns.distplot(word.map(lambda x: np.mean(x)), ax=ax1, color='red')
            ax1.set_title('Sarcastic Text')
            word = df[df['is_sarcastic'] == 0]['headline'].str.split().apply(lambda x: [len(i) for i in x])
            sns.distplot(word.map(lambda x: np.mean(x)), ax=ax2, color='green')
            ax2.set_title('Not Sarcastic Text')
            fig.suptitle('Average Word Length in Each Text')
            st.pyplot(fig)

    # Model Training and Evaluation
    st.header("Model Training and Evaluation")
    if uploaded_file is not None:
        model_option = st.selectbox("Choose your model",
                                    ['Logistic Regression', 'Naive Bayes',  'BertLSTM', 'BERTMLP',
                                     'RoBERTa'])

        if model_option in [ 'BertLSTM', 'BERTMLP', 'RoBERTa']:
            st.markdown("**Note:** Advanced models are demonstrated using pre-trained results.")

            # Replace 'path_to_your_screenshot.png' with the actual path or URL to your screenshot
            if model_option == 'BertLSTM':
                st.image("BERT+LSTM.png", caption="BERTLSTM Model Results")
            elif model_option == 'BERTMLP':
                st.image("BERT+MLP.png", caption="BERTMLP Model Results")
            elif model_option == 'RoBERTa':
                st.image("RoBerta.png", caption="RoBERTa Model Results")
        else:
            tfidf = TfidfVectorizer()
            if st.button("Train Model"):
                processed_data = process_text(df['headline'])
                labels = df['is_sarcastic']
                X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2,
                                                                    random_state=42)
                X_train = tfidf.fit_transform(X_train)
                X_test = tfidf.transform(X_test)

                if model_option == 'Logistic Regression':
                    clf = LogisticRegression(n_jobs=1, C=1e5)
                elif model_option == 'Naive Bayes':
                    clf = MultinomialNB()

                clf.fit(X_train, y_train)
                st.session_state['clf'] = clf  # Store the model in session state
                st.session_state['tfidf'] = tfidf  # Store the TFIDF vectorizer in session state

                train_accuracy, train_f1, test_accuracy, test_f1, report = evaluate_model(clf, X_train, y_train, X_test,
                                                                                          y_test)

                st.subheader("Model Performance Metrics:")
                st.write(f"Train Accuracy: {train_accuracy}")
                st.write(f"Test Accuracy: {test_accuracy}")
                st.write(f"Train F1 Score: {train_f1}")
                st.write(f"Test F1 Score: {test_f1}")
                st.text("Classification Report:")
                st.text(report)

                # Save the model
                filename = f'finalized_{model_option}_model.sav'
                pickle.dump(clf, open(filename, 'wb'))
                st.success(f"{model_option} model trained and saved successfully!")

    # LIME Explanations Section
    st.header("LIME Explanations for Model Predictions")
    if uploaded_file is not None and 'clf' in st.session_state and st.checkbox("Generate LIME Explanation"):
        clf = st.session_state['clf']  # Retrieve the model from session state
        tfidf = st.session_state['tfidf']  # Retrieve the TFIDF vectorizer from session state
        c = make_pipeline(tfidf, clf)  # Create the pipeline

        idx = np.random.randint(0, len(df))
        headline = df.iloc[idx]["headline"]
        explainer = LimeTextExplainer(class_names=["Non Sarcastic", "Sarcastic"])
        exp = explainer.explain_instance(headline, c.predict_proba, num_features=10)
        st.write(f"Example {idx}")
        st.write("Headline:", headline)
        st.write("True Class:", ["Non Sarcastic", "Sarcastic"][df.iloc[idx]["is_sarcastic"]])

        fig = exp.as_pyplot_figure()  # Use the as_pyplot_figure method directly
        st.pyplot(fig)

if __name__ == "__main__":
    main()
