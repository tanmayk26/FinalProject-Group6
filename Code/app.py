import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

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

# Main function for the Streamlit app
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: black;
    }
    .st-af {
        color: white;
        font-size: 20px;
    }
    .st-bb {
        background-color: grey;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Sarcasm Detection in Headlines")

    # Project Overview
    st.header("Project Overview")
    st.write("This project aims to detect sarcasm in news headlines using machine learning. It's an exploration into natural language processing and sentiment analysis.")

    # Sidebar with Project Info
    st.sidebar.header("Project Information")
    st.sidebar.info("""
        - **Project Title:** Sarcasm Detection in Headlines
        - **Author:** SSSS,TTTT,PPPP
        - **Institution:** Your University
        - **Department:** Your Department
    """)

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
            st.write(df.describe())
            st.subheader("Sarcasm Count in Dataset")
            sarcasm_count = df['is_sarcastic'].value_counts()
            st.bar_chart(sarcasm_count)

    # Model Training and Evaluation
    st.header("Model Training and Evaluation")
    if uploaded_file is not None:
        model_option = st.selectbox("Choose your model", ['Logistic Regression', 'Naive Bayes'])
        tfidf = TfidfVectorizer()
        if st.button("Train Model"):
            processed_data = process_text(df['headline'])
            labels = df['is_sarcastic']
            X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)
            X_train = tfidf.fit_transform(X_train)
            X_test = tfidf.transform(X_test)

            if model_option == 'Logistic Regression':
                clf = LogisticRegression(n_jobs=1, C=1e5)
            elif model_option == 'Naive Bayes':
                clf = MultinomialNB()

            clf.fit(X_train, y_train)
            train_accuracy, train_f1, test_accuracy, test_f1, report = evaluate_model(clf, X_train, y_train, X_test, y_test)

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
    if uploaded_file is not None and st.checkbox("Generate LIME Explanation"):
        idx = np.random.randint(0, len(df))
        headline = df.iloc[idx]["headline"]
        c = make_pipeline(tfidf, clf)
        class_names = ["Non Sarcastic", "Sarcastic"]
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(headline, c.predict_proba, num_features=10)
        st.write(f"Example {idx}")
        st.write("Headline:", headline)
        st.write("True Class:", class_names[df.iloc[idx]["is_sarcastic"]])
        fig, ax = plt.subplots()
        exp.as_pyplot_figure(ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()


# #import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from lime.lime_text import LimeTextExplainer
# from sklearn.pipeline import make_pipeline
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import pickle
#
# # Ensure the necessary NLTK downloads
# nltk.download('stopwords')
# nltk.download('wordnet')
#
# # Function to process text data
# def process_text(data):
#     lemma = nltk.WordNetLemmatizer()
#     stopwords_set = set(stopwords.words('english'))
#     processed_data = []
#     for sent in data:
#         words = [lemma.lemmatize(word) for word in sent.split() if word.lower() not in stopwords_set]
#         processed_data.append(' '.join(words))
#     return processed_data
#
# # Function to evaluate the model and return metrics
# def evaluate_model(clf, X_train, y_train, X_test, y_test):
#     y_train_pred = clf.predict(X_train)
#     y_test_pred = clf.predict(X_test)
#     train_accuracy = accuracy_score(y_train, y_train_pred)
#     test_accuracy = accuracy_score(y_test, y_test_pred)
#     train_f1 = f1_score(y_train, y_train_pred)
#     test_f1 = f1_score(y_test, y_test_pred)
#     report = classification_report(y_test, y_test_pred)
#     return train_accuracy, train_f1, test_accuracy, test_f1, report
#
# # Function to display a slide
# def show_slide(title, content_func):
#     st.markdown(f"## {title}")
#     content_func()
#
# # Slide contents
# def introduction():
#     st.write("This project aims to detect sarcasm in news headlines using machine learning.")
#
# def dataset_overview(uploaded_file):
#     if uploaded_file is not None:
#         df = pd.read_json(uploaded_file, lines=True)
#         st.dataframe(df.head())
#
# def data_analysis(uploaded_file):
#     if uploaded_file is not None:
#         df = pd.read_json(uploaded_file, lines=True)
#         st.write(df.describe())
#         sarcasm_count = df['is_sarcastic'].value_counts()
#         st.bar_chart(sarcasm_count)
#
# def methodology():
#     st.write("The project uses TfidfVectorizer for text processing and Logistic Regression/Naive Bayes for classification.")
#
# def results(uploaded_file, model_option):
#     if uploaded_file is not None:
#         df = pd.read_json(uploaded_file, lines=True)
#         processed_data = process_text(df['headline'])
#         labels = df['is_sarcastic']
#         X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)
#         tfidf = TfidfVectorizer()
#         X_train = tfidf.fit_transform(X_train)
#         X_test = tfidf.transform(X_test)
#
#         if model_option == 'Logistic Regression':
#             clf = LogisticRegression(n_jobs=1, C=1e5)
#         elif model_option == 'Naive Bayes':
#             clf = MultinomialNB()
#
#         clf.fit(X_train, y_train)
#         train_accuracy, train_f1, test_accuracy, test_f1, report = evaluate_model(clf, X_train, y_train, X_test, y_test)
#
#         st.write(f"Train Accuracy: {train_accuracy}")
#         st.write(f"Test Accuracy: {test_accuracy}")
#         st.write(f"Train F1 Score: {train_f1}")
#         st.write(f"Test F1 Score: {test_f1}")
#         st.text("Classification Report:")
#         st.text(report)
#
# def lime_explanation(uploaded_file, model_option):
#     if uploaded_file is not None:
#         df = pd.read_json(uploaded_file, lines=True)
#         processed_data = process_text(df['headline'])
#         labels = df['is_sarcastic']
#         X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)
#         tfidf = TfidfVectorizer()
#         X_train = tfidf.fit_transform(X_train)
#         X_test = tfidf.transform(X_test)
#
#         if model_option == 'Logistic Regression':
#             clf = LogisticRegression(n_jobs=1, C=1e5)
#         elif model_option == 'Naive Bayes':
#             clf = MultinomialNB()
#
#         clf.fit(X_train, y_train)
#
#         idx = np.random.randint(0, len(df))
#         headline = df.iloc[idx]["headline"]
#         c = make_pipeline(tfidf, clf)
#         class_names = ["Non Sarcastic", "Sarcastic"]
#         explainer = LimeTextExplainer(class_names=class_names)
#         exp = explainer.explain_instance(headline, c.predict_proba, num_features=10)
#         st.write("Example:")
#         st.write("Headline:", headline)
#         st.write("True Class:", class_names[df.iloc[idx]["is_sarcastic"]])
#         fig, ax = plt.subplots()
#         exp.as_pyplot_figure(ax=ax)
#         st.pyplot(fig)
#
# # Main function for the Streamlit app
# def main():
#     st.title("Sarcasm Detection in Headlines")
#
#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     slide_titles = ["Introduction", "Dataset Overview", "Data Analysis", "Methodology", "Results", "LIME Explanations"]
#     slide_index = st.sidebar.radio("Go to Slide", range(len(slide_titles)))
#
#     # File uploader for dataset
#     uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["json"])
#
#     # Model selection
#     model_option = st.sidebar.selectbox("Choose your model", ['Logistic Regression', 'Naive Bayes'])
#
#     # Show selected slide
#     if slide_index == 0:
#         show_slide(slide_titles[slide_index], introduction)
#     elif slide_index == 1:
#         show_slide(slide_titles[slide_index], lambda: dataset_overview(uploaded_file))
#     elif slide_index == 2:
#         show_slide(slide_titles[slide_index], lambda: data_analysis(uploaded_file))
#     elif slide_index == 3:
#         show_slide(slide_titles[slide_index], methodology)
#     elif slide_index == 4:
#         show_slide(slide_titles[slide_index], lambda: results(uploaded_file, model_option))
#     elif slide_index == 5:
#         show_slide(slide_titles[slide_index], lambda: lime_explanation(uploaded_file, model_option))
#
# if __name__ == "__main__":
#     main()
