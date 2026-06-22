import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

nltk.download('stopwords')

df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

def preprocess(text):
    text = text.lower()

    words = text.split()

    filtered = []

    for word in words:
        if word not in stopwords.words('english'):
            filtered.append(word)

    return " ".join(filtered)

df['message'] = df['message'].apply(preprocess)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['message'])

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully")