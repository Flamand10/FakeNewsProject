import nltk
import pandas as pd
import re
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure necessary downloads
nltk.download('stopwords')
nltk.download('punkt')

# Read CSV & process
def cleanF(file_path):
    df = pd.read_csv(file_path, dtype=str)
    
    # Ensure stopwords are lowercase
    stop_words = set(w.lower() for w in stopwords.words('english'))
    stemmer = PorterStemmer()

    # Vocab tracking
    original_vocab = set()
    filtered_vocab = set()
    stemmed_vocab = set()

    def clean_text(text):
        if not isinstance(text, str):
            text = str(text)

        # Tokenization (BETTER method than regex)
        word_tokens = word_tokenize(text)
        original_vocab.update(word_tokens)

        # Stopword removal (ignore non-alphabetic words)
        filtered_tokens = [w.lower() for w in word_tokens if w.isalpha() and w.lower() not in stop_words]
        filtered_vocab.update(filtered_tokens)

        # Stemming
        stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
        stemmed_vocab.update(stemmed_tokens)

        return " ".join(stemmed_tokens)

    # Apply cleaning function to each column
    cleaned_df = df.map(clean_text)

    # Calculate vocabulary sizes & reduction rates
    OG_vocab_size = len(original_vocab)
    filtered_vocab_size = len(filtered_vocab)
    stemmed_vocab_size = len(stemmed_vocab)
    stopword_reduction_rate = ((OG_vocab_size - filtered_vocab_size) / OG_vocab_size * 100) if OG_vocab_size else 0
    stemming_reduction_rate = ((filtered_vocab_size - stemmed_vocab_size) / filtered_vocab_size * 100) if filtered_vocab_size else 0

    # Print statistics
    print(f"Original Vocabulary Size: {OG_vocab_size}")
    print(f"Vocabulary Size After Stopword Removal: {filtered_vocab_size}")
    print(f"Reduction Rate After Stopword Removal: {stopword_reduction_rate:.2f}%")
    print(f"Vocabulary Size After Stemming: {stemmed_vocab_size}")
    print(f"Reduction Rate After Stemming: {stemming_reduction_rate:.2f}%")

    # Generate new file name correctly
    base_name, ext = os.path.splitext(file_path)
    new_file_name = f"{base_name}_cleaned{ext}"
    
    # Save cleaned data
    cleaned_df.to_csv(new_file_name, index=False)
    print(f" Successfully saved cleaned dataset: {new_file_name}")

# Example usage with multiple files
file_paths = ['news_sample.csv', '995,000_rows.csv']
for file_path in file_paths:
    cleanF(file_path)
