import pandas as pd

# Example dataset loading
df = pd.read_csv('social_media_sentiment.csv')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove special characters, URLs, mentions
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    # Lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Stopword removal and lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(preprocess_text)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df['sentiment']
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load pre-trained model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    cleaned_data = preprocess_text(data)
    vectorized_data = tfidf.transform([cleaned_data])
    prediction = model.predict(vectorized_data)
    
    return jsonify({'sentiment': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
