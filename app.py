import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk

# Load the models and vectorizers
with open('voting_classifier.pkl', 'rb') as f:
    voting_clf = pickle.load(f)

with open('logistic_reg_tfidf.pkl', 'rb') as f:
    _, tfidfvec = pickle.load(f), pickle.load(f)

def clean_text(text):
    import re
    from bs4 import BeautifulSoup
    from nltk.stem import PorterStemmer
    lemmatizer = nltk.WordNetLemmatizer()
    nltk.download('wordnet')
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    clean_text = re.sub(r'[^A-Za-z0-9\s]+', '', clean_text)
    words = clean_text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

st.title("Sentiment Analysis App")
user_input = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    cleaned_input = clean_text(user_input)
    vectorized_input = tfidfvec.transform([cleaned_input])
    prediction = voting_clf.predict(vectorized_input)

    if prediction == 1:
        st.success("The sentiment is Positive")
    else:
        st.error("The sentiment is Negative")
