import nltk
import pandas as pd
import re
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Limit CPU usage to prevent overheating
os.environ["OMP_NUM_THREADS"] = "2"

nltk.download('stopwords')

#nltk stopwords
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """ Cleans a single text entry. """
    if pd.isna(text):
        return ""  # Handle missing values

    # Fix unicode issues
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # Tokenize without punctuation, only words
    word_tokens = re.findall(r'\b\w+\b', text)

    # Remove stopwords (ensure they are lowercased)
    filtered_tokens = [w.lower() for w in word_tokens if w.lower() not in stop_words]

    # Apply stemming
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]

    return " ".join(stemmed_tokens)  # Return cleaned text

def cleanF_chunked(filename, output_filename, chunk_size=100000):
    """ Cleans a large dataset in chunks and processes each row correctly. """

    OG_vocab_size = 0
    filtered_vocab_size = 0
    stemmed_vocab_size = 0

    cleaned_data = []

    # Read large file in chunks
    for chunk in pd.read_csv(filename, dtype=str, encoding='utf-8', chunksize=chunk_size):
        
        # Apply cleaning function to each column separately
        for column in chunk.columns:
            chunk[column] = chunk[column].astype(str).map(clean_text)  # Replaces applymap()

        # Compute vocabulary size per chunk
        text = " ".join(chunk.astype(str).agg(" ".join, axis=1))
        word_tokens = re.findall(r'\b\w+\b', text)

        OG_vocab_size += len(set(word_tokens))

        filtered_tokens = [w.lower() for w in word_tokens if w.lower() not in stop_words]
        filtered_vocab_size += len(set(filtered_tokens))

        stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
        stemmed_vocab_size += len(set(stemmed_tokens))

        cleaned_data.append(chunk)

    # Compute reduction rates (avoid division by zero)
    stopword_reduction_rate = ((OG_vocab_size - filtered_vocab_size) / OG_vocab_size * 100) if OG_vocab_size > 0 else 0
    stemming_reduction_rate = ((filtered_vocab_size - stemmed_vocab_size) / filtered_vocab_size * 100) if filtered_vocab_size > 0 else 0

    # Print statistics ONCE per file
    print(f"\nProcessing: {filename}")
    print(f"Original Vocabulary Size: {OG_vocab_size}")
    print(f"Vocabulary Size After Stopword Removal: {filtered_vocab_size}")
    print(f"Reduction Rate After Stopword Removal: {stopword_reduction_rate:.2f}%")
    print(f"Vocabulary Size After Stemming: {stemmed_vocab_size}")
    print(f"Reduction Rate After Stemming: {stemming_reduction_rate:.2f}%")

    # Concatenate all cleaned chunks and save
    final_df = pd.concat(cleaned_data, ignore_index=True)
    final_df.to_csv(output_filename, index=False, encoding='utf-8')

    print(f"\n Successfully saved cleaned dataset: {output_filename}")

# Process small dataset
cleanF_chunked('news_sample.csv', 'small_sample_reduced.csv')

# Process large dataset in chunks
cleanF_chunked('995,000_rows.csv', 'news_sample_reduced.csv')
