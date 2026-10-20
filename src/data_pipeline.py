import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

def clean_text_basic(text):
    """Basic text cleaning for NLP pipeline."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text.strip()

if __name__ == "__main__":
    print("Initializing NLP Pipeline...")
    sample_data = pd.DataFrame({
        'ticket_id': [1, 2],
        'customer_text': [
            "My internet is NOT working! Fix it ASAP http://link.com",
            "I loved the quick service, it was better than expected."
        ]
    })

    print("Original Data:\n", sample_data['customer_text'].values)
    sample_data['cleaned_text'] = sample_data['customer_text'].apply(clean_text_basic)

    print("\nExtracting Features using TF-IDF...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sample_data['cleaned_text'])
    print(f"Feature matrix shape: {tfidf_matrix.shape}")