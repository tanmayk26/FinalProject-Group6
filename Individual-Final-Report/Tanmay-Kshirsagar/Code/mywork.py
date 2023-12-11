######### scrapper.py
import pandas as pd
import os
import bs4
from tqdm import tqdm
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from socket import gaierror
import gdown

def download_available_data(target_dir):
    current_directory = os.getcwd()
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    os.chdir(target_dir)
    file_id_mapping = {
        "Sarcasm_Headlines_Dataset.json": "1NSpIABIfwZ3E1As6XVbNCX3do0D9MucI",
        "sarcastic_output.json": "19dS1iQ51oxRmiEkoArWUwW6BqXYDJGuo"
    }
    for key, value in file_id_mapping.items():
        if not os.path.exists(os.path.join(target_dir, key)):
            downloaded_file = gdown.download(id=value, quiet=False)
            if downloaded_file != key:
                print(f"{key} could not be downloaded")
    os.chdir(current_directory)


def extract_text_from_url(url):
    offset = 30
    try:
        source = urlopen(url).read()
        soup = bs4.BeautifulSoup(source, 'lxml')
    except (URLError, gaierror):
        try:
            new_url = url[offset:]
            source = urlopen(new_url).read()
            soup = bs4.BeautifulSoup(source, 'lxml')
        except (URLError, gaierror, ValueError, HTTPError):
            text = ""
            return text
    allowlist = ["p", "em", "i", "b"]
    blocklist = ["Sign Up", "HuffPost", "Huffington"]
    text_elements = list()
    for t in soup.find_all(text=True):
        if t.parent.name in allowlist:
            contains_blocked = False
            for block in blocklist:
                if t.find(block) != -1:
                    contains_blocked = True
            if not contains_blocked:
                text_elements.append(t)

    text = " ".join(text_elements)
    return text


code_directory = os.getcwd()
data_directory = os.path.join(os.path.split(code_directory)[0], 'Code')
download_available_data(data_directory)
df1 = pd.read_json(os.path.join(data_directory, r'Sarcasm_Headlines_Dataset.json'), lines=True)
df2 = pd.read_json(os.path.join(data_directory, r'Sarcasm_Headlines_Dataset_v2.json'), lines=True)
df2 = df2[['article_link', 'headline', 'is_sarcastic']]
final_data = pd.concat([df1, df2], ignore_index=True)
sarcastic_data = final_data.loc[final_data['is_sarcastic'] == 1]
sarcastic_data.reset_index(drop=True, inplace=True)
print(sarcastic_data.head())
print(sarcastic_data.columns)
print(sarcastic_data.shape)

sarcastic_data["body"] = [""] * len(sarcastic_data)
print("\nScraping TheOnion articles...")
with tqdm(total=len(sarcastic_data)) as progress_bar:
    for i, row in sarcastic_data.iterrows():
        body = extract_text_from_url(row[0])
        sarcastic_data.loc[i, "body"] = body
        progress_bar.update()

output_path = os.path.join(data_directory, "sarcastic_news_text.json")
sarcastic_data.to_json(output_path)

######### text_summarization.py
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import sent_tokenize
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_scheduler, pipeline
from accelerate import Accelerator
import evaluate
from tqdm import tqdm

# Constants
data_splits = {'train': 0.8, 'val': 0.1, 'test': 0.1}
input_var = 'body'
target_var = 'headline'
word_limit = 30
model_checkpoint = "JulesBelveze/t5-small-headline-generator"
TRAIN_MODEL = True
random_seed = 42
batch_size = 16
num_train_epochs = 30

# Directories
code_dir = os.getcwd()
data_dir = os.path.join(os.path.split(code_dir)[0], 'Code')
model_dir = os.path.join(os.path.split(code_dir)[0], 'Code')

# Function to filter training indices
def filter_indices(data):
    return np.array([len(x) > word_limit for x in data.body.str.split()])

# Function to display samples
def show_samples(data, num_samples=3, seed=random_seed):
    sample_data = data["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample_data:
        print(f"\n'>> Headline: {example[target_var]}'")
        print(f"'>> Body: {example[input_var]}'")

# Function for one-sentence summary
def generate_summary(text):
    return "\n".join(sent_tokenize(text)[:1])

# Function to evaluate baseline
def evaluate_baseline(data, metric):
    summaries = [generate_summary(text) for text in data[input_var]]
    return metric.compute(predictions=summaries, references=data[target_var])

# Function for post-processing text
def postprocess(predictions, labels):
    predictions = [pred.strip() for pred in predictions]
    labels = [label.strip() for label in labels]

    predictions = ["\n".join(sent_tokenize(pred)) for pred in predictions]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    return predictions, labels

# Function to preprocess data
def preprocess(examples):
    model_inputs = tokenizer(
        examples[input_var],
        max_length=max_input_length,
        truncation=True
    )
    label = tokenizer(
        examples[target_var], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = label["input_ids"]
    return model_inputs

# Function to get a summary
def get_summary(index):
    reviews = dataset['test'][index]["body"]
    titles = dataset["test"][index]["headline"]
    summaries = summarizer(dataset["test"][index]["body"], min_length=10, max_length=max_target_length)[0]["summary_text"]
    return reviews, titles, summaries

# Loading data
df = pd.read_json(os.path.join(data_dir, 'sarcastic_news_text.json'))
df = df[[input_var, target_var]]
train_val_df = df[filter_indices(df)]
test_df = df[~filter_indices(df)]
max_input_length = int(np.percentile([len(x) for x in train_val_df[input_var].str.split()], 99.5))
max_target_length = int(np.percentile([len(x) for x in train_val_df[target_var].str.split()], 99.5))

# Dataset creation
dataset = DatasetDict()
train_val_df = train_val_df.sample(frac=1, random_state=random_seed).reset_index()
total_percentage = 0
for split, percentage in data_splits.items():
    index_range = (int(total_percentage * len(train_val_df)), int((total_percentage + percentage) * len(train_val_df)))
    total_percentage = total_percentage + percentage
    dataset[split] = Dataset.from_pandas(train_val_df.iloc[index_range[0]:index_range[1]])

print("\nSample training data")
show_samples(dataset)

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print("\nTokenizing datasets")
tokenized_datasets = dataset.map(preprocess, batched=True)

# Lead-1 baseline
rouge_score = evaluate.load("rouge")
baseline_score = evaluate_baseline(dataset["val"], rouge_score)
print(f"\nBaseline score on validation dataset is: ")
for score, value in baseline_score.items():
    print(score, ": ", value)


# Lists to store Rouge scores for plotting
rouge_keys = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
rouge_scores = {key: [] for key in rouge_keys}

# Training the summarizer
if TRAIN_MODEL:
    print('Training')
    lr = 2e-5
    # Fine-tuning
    tokenized_datasets.set_format("torch")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)
    features = [tokenized_datasets["train"][i] for i in range(2)]
    data_collator(features)
    # DataLoader definition
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["val"], collate_fn=data_collator, batch_size=batch_size
    )
    # Optimizer initialization
    optimizer = AdamW(model.parameters(), lr=lr)
    # Accelerator preparation
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    # Learning rate schedule definition
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    # Model and optimizer saving directory
    output_directory = os.path.join(model_dir, "t5-sarcastic-headline-generator")
    progress_bar = tqdm(range(num_training_steps))
    print("\nFine-tuning pretrained model")
    for epoch in range(num_train_epochs):
        model.train()
        progress_bar.set_description(f"Epoch: [{epoch + 1}/{num_train_epochs}]")
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                )
                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess(
                    decoded_preds, decoded_labels
                )
                rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

        result = rouge_score.compute()
        result = {k: round(v, 4) for k, v in result.items()}

        # Append Rouge scores to the lists
        for key in rouge_keys:
            rouge_scores[key].append(result[key])

        print(f"Epoch {epoch + 1}:", end=" ")
        for key, value in result.items():
            print(key, ' : ', value, end=" ")
        print(" ")

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_directory, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_directory)


# Predictions on test data after loading model
saved_model = os.path.join(model_dir, "t5-sarcastic-headline-generator")
summarizer = pipeline("summarization", model=saved_model)
# Plotting the Rouge scores
plt.figure(figsize=(10, 6))
for key in rouge_keys:
    plt.plot(range(1, num_train_epochs + 1), rouge_scores[key], label=key)

plt.xlabel('Epoch')
plt.ylabel('Rouge Score')
plt.title('Epoch vs Rouge Score')
plt.legend()
plt.show()
# Print 3 random predictions
prediction_index = []
counter = 0
while counter < 3:
    flag = False
    index = random.randint(0, len(dataset['test']))
    if index not in prediction_index:
        flag = True
        counter = counter + 1
        prediction_index.append(index)
    if flag:
        review, title, summary = get_summary(index)
        print(f'\n-- Prediction {counter} --')
        print(f"'>>> Article: {review}'")
        print(f"\n'>>> Headline: {title}'")
        print(f"\n'>>> Summary: {summary}'")

print("\n Building Predictions.csv for all test data")
main_list = []
for i in tqdm(range(len(dataset['test']))):
    review, title, summary = get_summary(i)
    temp = [review, title, summary]
    main_list.append(temp)

# Save predictions into csv file
predictions_df = pd.DataFrame(main_list, columns=['Body', 'Headline', 'Summary'])
predictions_df.to_csv(os.path.join(data_dir, 'Summary_Predictions.csv'), index=False)

######### text_summarization.py
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Config
import matplotlib.pyplot as plt
import shap
saved_model = "/content/sample_data/t5-sarcastic-headline-generator"
summarizer = pipeline("summarization", model=saved_model)
article = """Following the release of a report indicating that the agency failed 95 percent of security tests, the Transportation Security Administration announced Tuesday that agents will now simply stand at airport checkpoints and remind all passengers that everybody will eventually die someday. “As part of our new security protocol, TSA agents at every checkpoint will carefully inform each passenger that life is a temporary state and that no man can escape the fate that awaits us all,” said acting TSA administrator Mark Hatfield, adding that under the new guidelines, agents will ensure that passengers fully understand and accept the inevitability of death as they proceed through the boarding pass check, luggage screening, and body scanner machines. “Signs posted throughout the queues will also state that death is unpredictable but guaranteed, and a series of looping PA messages will reiterate to passengers that, even if they survive this flight, they could still easily die in 10 years or even tomorrow.” Hatfield went on to say that the TSA plans to add a precheck program that will expedite the process for passengers the agency deems comfortable with the ephemeral nature of life.'"""
summ = summarizer(article, min_length=10, max_length=30)[0]['summary_text']
print(f'\n-- Prediction --')
print(f"'Article: {article}'")
print(f"\n'Summary: {summ}'")
tokenizer = AutoTokenizer.from_pretrained(saved_model)
model_config = T5Config.from_pretrained(saved_model)
model = T5ForConditionalGeneration.from_pretrained(saved_model, config=model_config)
ip_text = [article]
explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(ip_text)
print(shap_values.feature_names)
shap.plots.text(shap_values)
### app.py
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.pipeline import make_pipeline
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


# Function for sarcasm prediction
def predict_sarcasm(text, clf, tfidf):
    processed_text = process_text([text])
    tfidf_text = tfidf.transform(processed_text)
    prediction = clf.predict(tfidf_text)
    return "Sarcastic" if prediction[0] == 1 else "Non Sarcastic"


# Function for text summarization
def text_summarization(text):
    saved_model = "t5-sarcastic-headline-generator"
    summarizer = pipeline("summarization", model=saved_model)
    summarized_text = summarizer(text, min_length=10, max_length=30)[0]['summary_text']
    return summarized_text

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


    # Model Training and Evaluation
    st.header("Model Training and Evaluation")
    if uploaded_file is not None:
        model_option = st.selectbox("Choose your model",
                                    ['Logistic Regression', 'Naive Bayes', 'LSTM', 'BERT', 'BertLSTM', 'BERTMLP',
                                     'RoBERTa', 'Text Summarization'])

        if model_option in ['LSTM', 'BERT', 'BertLSTM', 'BERTMLP', 'RoBERTa', 'Text Summarization']:
            st.markdown("**Note:** Advanced models are demonstrated using pre-trained results.")

            # Replace 'path_to_your_screenshot.png' with the actual path or URL to your screenshot
            if model_option == 'BERT':
                st.image("Picture1.png", caption="BERT Model Results")
            elif model_option == 'LSTM':
                st.image("lstm.png", caption="LSTM Model Results")
            elif model_option == 'BertLSTM':
                st.image("BERT+LSTM.png", caption="BERTLSTM Model Results")
            elif model_option == 'BERTMLP':
                st.image("BERT+MLP.png", caption="BERTMLP Model Results")
            elif model_option == 'RoBERTa':
                st.image("RoBerta.png", caption="RoBERTa Model Results")
            elif model_option == 'Text Summarization':
                st.image("summ_rouge_vs_epoch.png", caption="Text Summarization Model Results")
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


    # Real-time Text Summarization
    st.header("Real-time Text Summarization")
    user_input_summarization = st.text_area("Enter text to summarize")
    if user_input_summarization:
        summarized_text = text_summarization(user_input_summarization)
        st.subheader("Summarized Text:")
        st.write(summarized_text)

    # Real-time Sarcasm Prediction
    st.header("Real-time Sarcasm Prediction")
    user_input_sarcasm = st.text_input("Enter a headline to predict sarcasm")
    if user_input_sarcasm:
        if 'clf' in st.session_state:
            prediction_result = predict_sarcasm(user_input_sarcasm, st.session_state['clf'], st.session_state['tfidf'])
            st.write(f"Prediction: **{prediction_result}**")

if __name__ == "__main__":
    main()