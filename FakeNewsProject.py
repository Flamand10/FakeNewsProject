import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
# nltk.download('stopwords') for at det virker


#read csv

def cleanF(file_path):
    df = pd.read_csv(file_path, dtype=str)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    original_vocab = set()
    filtered_vocab = set()
    stemmed_vocab = set()

    def clean_text(text):
        if not isinstance(text, str):
            text = str(text)
        # Tokenize without punctuation, only words
        word_tokens = re.findall(r'\b\w+\b', text)
        original_vocab.update(word_tokens)
        # Remove stopwords
        filtered_tokens = [w for w in word_tokens if w.lower() not in stop_words]
        filtered_vocab.update(filtered_tokens)
        # Stemming
        stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
        stemmed_vocab.update(stemmed_tokens)
        return " ".join(stemmed_tokens)

    # Apply cleaning to each column
    cleaned_df = df.applymap(clean_text)

    # Calculate vocabulary sizes and reduction rates
    OG_vocab_size = len(original_vocab)
    filtered_vocab_size = len(filtered_vocab)
    stemmed_vocab_size = len(stemmed_vocab)
    stopword_reduction_rate = ((OG_vocab_size - filtered_vocab_size) / OG_vocab_size * 100)
    stemming_reduction_rate = ((filtered_vocab_size - stemmed_vocab_size) / filtered_vocab_size * 100)

    # Print statistics
    print(f"Original Vocabulary Size: {OG_vocab_size}")
    print(f"Vocabulary Size After Stopword Removal: {filtered_vocab_size}")
    print(f"Reduction Rate After Stopword Removal: {stopword_reduction_rate:.2f}%")
    print(f"Vocabulary Size After Stemming: {stemmed_vocab_size}")
    print(f"Reduction Rate After Stemming: {stemming_reduction_rate:.2f}%")

    # Generate new file name with 'cleaned' added to the current CSV name
    base_name, ext = os.path.splitext(file_path)
    new_file_name = f"{base_name}_cleaned{ext}"
    
    cleaned_df.to_csv(new_file_name, index=False)

# Example usage with multiple files
file_paths = ['news_sample.csv', '995,000_rows.csv']
for file_path in file_paths:
    cleanF(file_path)
