import re
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk
"""
Stemming is a text preprocessing technique that reduces words to their root
or base form by chopping off prefixes or suffixes.
The goal is to treat different forms of a word as the same token,
simplifying the vocabulary and helping models generalize better.

Example of stemming:
Original Word	Stemmed Word
playing	play
played	play
"""
nltk.download('punkt')

stemmer = PorterStemmer()

def clean_and_stem_tokenizer(text):
    # 1. Lowercase the text
    text = text.lower()

    # 2. Remove non-alphabetic characters using regex
    #    This keeps spaces and alphabet letters only
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Tokenize the cleaned text
    tokens = word_tokenize(text)

    # 4. Stem each token
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

data = {
    'Text': [
        "I loved learning about machines!",
        "Learning machines is GREAT!!!",
        "I hate bugs in code...",
        "Debugging is fun :)",
        "Bugs make coding frustrating!"
    ],
    'Sentiment': [1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
X = df['Text']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=clean_and_stem_tokenizer, stop_words='english')),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
