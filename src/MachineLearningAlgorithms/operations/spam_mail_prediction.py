"""
Spam Mail Prediction using TF-IDF (with stopwords and min_df=1)
- End-to-end: load data -> clean text -> TF-IDF -> train model -> evaluate -> predict on new email
- Heavily commented for clarity.
"""

import os
import re
import pandas as pd
from typing import Tuple

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# -----------------------------
# 1) TEXT CLEANING
# -----------------------------
def clean_text(text: str) -> str:
    """
    Purpose:
        Normalize/clean raw text to reduce noise before vectorization.

    What it does:
        - Lowercase to make tokens case-insensitive.
        - Replace non-word chars with space (so punctuation/symbols don't become tokens).
        - Collapse multiple spaces to a single space for tidier tokens.
    """
    text = text.lower()
    text = re.sub(r"\W", " ", text)      # keep only letters/digits/underscore as token chars
    text = re.sub(r"\s+", " ", text)     # collapse repeated whitespace
    return text.strip()


# -----------------------------
# 2) DATA LOADING
# -----------------------------
def load_dataset() -> pd.DataFrame:
    """
    Try to load the classic SMS Spam Collection (spam.csv) if it's available.
    If not found, fall back to a tiny built-in dataset so the script still runs.

    Expected CSV format:
        - Column 'v1' -> label ('ham' or 'spam')
        - Column 'v2' -> message text
    """
    if os.path.exists("spam.csv"):
        df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "message"]
        print("Loaded dataset from spam.csv")
    else:
        print("spam.csv not found. Using a small built-in sample dataset for demonstration.")
        samples = {
            "label": [
                "ham", "ham", "ham", "ham",
                "spam", "spam", "spam", "spam"
            ],
            "message": [
                "Hey are we still meeting at 3 pm for the project?",
                "Don't forget to bring the documents for review.",
                "Can we move the call to tomorrow morning?",
                "Lunch today at the cafeteria?",
                "Congratulations! You have won a $1000 Walmart gift card. Click here to claim.",
                "URGENT! Your account has been compromised. Verify your details immediately.",
                "You have been selected for a free vacation. Reply YES to claim now!",
                "Win cash now!!! Visit our site and enter your details."
            ],
        }
        df = pd.DataFrame(samples)

    # Map labels -> numeric (ham=0, spam=1) for classification
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    # Clean messages
    df["message"] = df["message"].apply(clean_text)
    return df


# -----------------------------
# 3) TRAIN / TEST SPLIT
# -----------------------------
def split_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split text and labels into train/test sets.
    - stratify=y preserves class ratio (important for imbalanced spam datasets)
    - random_state ensures reproducibility
    """
    X = df["message"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() == 2 else None  # safe if tiny sample
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# -----------------------------
# 4) VECTORIZE TEXT WITH TF-IDF
# -----------------------------
def build_vectorizer() -> TfidfVectorizer:
    """
    Configure TF-IDF to:
      - remove English stop words (common words that carry little meaning)
      - min_df=1  (keep any term that appears in at least 1 document)
      - max_features=5000  (cap vocabulary size for efficiency; tweak as you like)

    NOTE: You can experiment with n-grams:
      - e.g., ngram_range=(1,2) to include unigrams + bigrams.
    """
    return TfidfVectorizer(
        stop_words="english",
        min_df=1,
        max_features=5000,
        # ngram_range=(1, 2),   # uncomment to include bigrams
    )


# -----------------------------
# 5) TRAIN MODEL
# -----------------------------
def train_model(X_train_tfidf, y_train) -> LogisticRegression:
    """
    Train a simple, strong baseline classifier for text: Logistic Regression.
    - max_iter bumped up to ensure convergence on larger corpora.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    return model


# -----------------------------
# 6) EVALUATION
# -----------------------------
def evaluate(model, X_test_tfidf, y_test) -> None:
    """
    Print standard metrics to understand performance:
      - Accuracy: overall correctness
      - Confusion Matrix: breakdown of TP/FP/FN/TN
      - Classification Report: precision, recall, f1 per class
    """
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))


# -----------------------------
# 7) PREDICT ON A NEW EMAIL
# -----------------------------
def predict_mail(mail_text: str, vectorizer: TfidfVectorizer, model: LogisticRegression) -> Tuple[str, float]:
    """
    Take a raw email string, clean it, vectorize it using the *fitted* TF-IDF, and predict.

    Returns:
        label_str: "Spam" or "Ham"
        confidence: probability of the predicted class (0..100%)
    """
    # Clean the incoming email text the same way we cleaned training data
    cleaned = clean_text(mail_text)

    # Transform with the *already fitted* vectorizer
    vec = vectorizer.transform([cleaned])

    # Predict class and get confidence (probability)
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][pred]

    label_str = "Spam" if pred == 1 else "Ham"
    confidence = round(float(proba) * 100.0, 2)
    return label_str, confidence


# -----------------------------
# 8) MAIN: WIRE IT ALL TOGETHER
# -----------------------------
if __name__ == "__main__":
    # Load and prepare data
    df = load_dataset()

    # Split into train/test
    X_train, X_test, y_train, y_test = split_data(df)

    # Fit TF-IDF on training text only (avoid test leakage)
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train classifier
    model = train_model(X_train_tfidf, y_train)

    # Evaluate on hold-out test set
    evaluate(model, X_test_tfidf, y_test)

    # Demo predictions on new emails
    print("\n=== New Email Predictions ===")
    examples = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now.",
        "Hi Sarah, can you send me the latest project report before our meeting tomorrow?",
        "URGENT: Your bank account is locked. Verify your identity immediately.",
        "Team, lunch at 12:30? I booked a table downstairs."
    ]
    for msg in examples:
        label, conf = predict_mail(msg, vectorizer, model)
        print(f"\nEmail: {msg}\nPrediction: {label} ({conf}% confidence)")
