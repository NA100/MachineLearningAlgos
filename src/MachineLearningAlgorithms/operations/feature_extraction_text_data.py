from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

"""
How it works
TF (Term Frequency) → How often a word appears in a document.
TF(t, d) = \frac{\text{# of times term t appears in document d}}{\text{total terms in d}}
IDF (Inverse Document Frequency) → How unique that word is across all documents.
IDF(t) = \log \frac{\text{# total documents}}{\text{# documents containing term t}}
TF-IDF → Multiply TF and IDF.
* High TF-IDF = word appears often in a document but rarely in others → important word.
* Low TF-IDF = word is common in many documents (e.g., "the", "and") → less important.

Where it’s used
* Text classification (spam detection, sentiment analysis)
* Search engines
* Keyword extraction
"""

# Sample text data
documents = [
    "I love machine learning",
    "Machine learning is great",
    "Deep learning is a branch of machine learning"
]

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert to DataFrame for readability
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print(df_tfidf)
