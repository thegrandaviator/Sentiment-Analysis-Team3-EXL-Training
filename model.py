import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
import pickle

# Load the dataset
IMDB_data = pd.read_csv("C:\Sentiment_Analysis2\IMDB Dataset.csv")  # Replace with the correct path to your dataset

# Preprocessing functions
nltk.download("stopwords")
lemmatizer = nltk.WordNetLemmatizer()
nltk.download('wordnet')
stopwords_l = stopwords.words("english")
token = ToktokTokenizer()

def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    clean_text = re.sub(r'[^A-Za-z0-9\s]+', '', clean_text)
    words = clean_text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Clean the reviews
IMDB_data['review'] = IMDB_data['review'].apply(remove_html)

# Encode sentiment column to binary (positive=1, negative=0)
label_encoder = LabelBinarizer()
IMDB_data['sentiment'] = label_encoder.fit_transform(IMDB_data['sentiment'])

# Split dataset into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(IMDB_data['review'], IMDB_data['sentiment'], test_size=0.2, random_state=42)

# Logistic Regression using CountVectorizer
countVec = CountVectorizer(min_df=0.0, max_df=1, binary=False, ngram_range=(1, 3))
cv_train = countVec.fit_transform(train_X)
cv_test = countVec.transform(test_X)

log_reg_cv = LogisticRegression(max_iter=1000)
log_reg_cv.fit(cv_train, train_Y)

# Save Logistic Regression model and CountVectorizer
with open('logistic_reg_countvec.pkl', 'wb') as f:
    pickle.dump(log_reg_cv, f)
    pickle.dump(countVec, f)
print("Logistic Regression with CountVectorizer model saved.")

# Logistic Regression using TfidfVectorizer
tfidfvec = TfidfVectorizer(min_df=0.0, max_df=1, binary=False, ngram_range=(1, 3))
tfidf_train = tfidfvec.fit_transform(train_X)
tfidf_test = tfidfvec.transform(test_X)

log_reg_tfidf = LogisticRegression(max_iter=1000)
log_reg_tfidf.fit(tfidf_train, train_Y)

# Save Logistic Regression model and TfidfVectorizer
with open('logistic_reg_tfidf.pkl', 'wb') as f:
    pickle.dump(log_reg_tfidf, f)
    pickle.dump(tfidfvec, f)
print("Logistic Regression with TfidfVectorizer model saved.")

# Multinomial Naive Bayes using CountVectorizer
nb_model_cv = MultinomialNB()
nb_model_cv.fit(cv_train, train_Y)

# Save Naive Bayes model and CountVectorizer
with open('nb_model_countvec.pkl', 'wb') as f:
    pickle.dump(nb_model_cv, f)
    pickle.dump(countVec, f)
print("Naive Bayes with CountVectorizer model saved.")

# Multinomial Naive Bayes using TfidfVectorizer
nb_model_tfidf = MultinomialNB()
nb_model_tfidf.fit(tfidf_train, train_Y)

# Save Naive Bayes model and TfidfVectorizer
with open('nb_model_tfidf.pkl', 'wb') as f:
    pickle.dump(nb_model_tfidf, f)
    pickle.dump(tfidfvec, f)
print("Naive Bayes with TfidfVectorizer model saved.")

# Voting Classifier using Logistic Regression and Naive Bayes
voting_clf = VotingClassifier(estimators=[
    ('log_reg_cv', log_reg_cv),
    ('log_reg_tfidf', log_reg_tfidf),
    ('nb_model_cv', nb_model_cv),
    ('nb_model_tfidf', nb_model_tfidf)
], voting='hard')
voting_clf.fit(tfidf_train, train_Y)

# Save Voting Classifier model
with open('voting_classifier.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)
print("Voting Classifier model saved.")

# Model Evaluation Function
def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return accuracy, report

# Evaluate Logistic Regression with CountVectorizer
log_reg_accuracy, log_reg_report = evaluate_model(log_reg_cv, cv_test, test_Y)
print(f"Logistic Regression with CountVectorizer Accuracy: {log_reg_accuracy}")
print(log_reg_report)

# Evaluate Logistic Regression with TfidfVectorizer
log_reg_tfidf_accuracy, log_reg_tfidf_report = evaluate_model(log_reg_tfidf, tfidf_test, test_Y)
print(f"Logistic Regression with TfidfVectorizer Accuracy: {log_reg_tfidf_accuracy}")
print(log_reg_tfidf_report)

# Evaluate Naive Bayes with CountVectorizer
nb_accuracy, nb_report = evaluate_model(nb_model_cv, cv_test, test_Y)
print(f"Naive Bayes with CountVectorizer Accuracy: {nb_accuracy}")
print(nb_report)

# Evaluate Naive Bayes with TfidfVectorizer
nb_tfidf_accuracy, nb_tfidf_report = evaluate_model(nb_model_tfidf, tfidf_test, test_Y)
print(f"Naive Bayes with TfidfVectorizer Accuracy: {nb_tfidf_accuracy}")
print(nb_tfidf_report)

# Evaluate Voting Classifier
voting_accuracy, voting_report = evaluate_model(voting_clf, tfidf_test, test_Y)
print(f"Voting Classifier Accuracy: {voting_accuracy}")
print(voting_report)
