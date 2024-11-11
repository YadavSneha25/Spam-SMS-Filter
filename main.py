import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
sms_raw = pd.read_csv("sms_spam.csv")

# Convert the 'type' column to categorical (spam or ham)
sms_raw['type'] = sms_raw['type'].astype('category')

# Preprocess function: Convert counts to "Yes" or "No"
def convert_counts(x):
    return "Yes" if x > 0 else "No"

# Preprocess the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove numbers and punctuation
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    return text

# Apply preprocessing to the text column
sms_raw['clean_text'] = sms_raw['text'].apply(preprocess_text)

# Split the data into train and test sets
sms_train, sms_test, sms_train_labels, sms_test_labels = train_test_split(
    sms_raw['clean_text'], sms_raw['type'], test_size=0.2, random_state=42)

# Vectorize the text data (CountVectorizer will create a document-term matrix)
vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
X_train = vectorizer.fit_transform(sms_train)
X_test = vectorizer.transform(sms_test)

# Train a Naive Bayes model
nb_classifier = MultinomialNB(alpha=1)
nb_classifier.fit(X_train, sms_train_labels)

# Predict on the test data
sms_test_predict = nb_classifier.predict(X_test)

# Evaluate the model
print(classification_report(sms_test_labels, sms_test_predict))

# Confusion matrix
conf_matrix = confusion_matrix(sms_test_labels, sms_test_predict)
print("Confusion Matrix:")
print(conf_matrix)

# Visualize word cloud for spam and ham
spam_text = " ".join(sms_raw[sms_raw['type'] == 'spam']['clean_text'])
ham_text = " ".join(sms_raw[sms_raw['type'] == 'ham']['clean_text'])

# Wordcloud for spam
wordcloud_spam = WordCloud(max_words=100, background_color="white").generate(spam_text)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud_spam, interpolation="bilinear")
plt.title("Spam Word Cloud")
plt.axis('off')
plt.show()

# Wordcloud for ham
wordcloud_ham = WordCloud(max_words=100, background_color="white").generate(ham_text)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud_ham, interpolation="bilinear")
plt.title("Ham Word Cloud")
plt.axis('off')
plt.show()

# Wordcloud for all cleaned text
wordcloud_all = WordCloud(max_words=100, background_color="white").generate(" ".join(sms_raw['clean_text']))
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud_all, interpolation="bilinear")
plt.title("All SMS Word Cloud")
plt.axis('off')
plt.show()

# Reducing number of features by finding frequent words
# Using CountVectorizer's feature importance based on frequency
vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'), max_df=0.95, min_df=5)
X_train = vectorizer.fit_transform(sms_train)
X_test = vectorizer.transform(sms_test)

# Train Naive Bayes model again with reduced features
nb_classifier = MultinomialNB(alpha=1)
nb_classifier.fit(X_train, sms_train_labels)

# Predict again
sms_test_predict = nb_classifier.predict(X_test)

# Evaluate the model
print("Updated Classification Report:")
print(classification_report(sms_test_labels, sms_test_predict))

# Confusion matrix
conf_matrix = confusion_matrix(sms_test_labels, sms_test_predict)
print("Updated Confusion Matrix:")
print(conf_matrix)
