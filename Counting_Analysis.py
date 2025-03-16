import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import re


# File paths
original_file = "995,000_rows.csv"
cleaned_file = "995,000_rows_cleaned.csv"

# Chunk size for reading large files
chunk_size = 10000  

# Function to process a CSV file efficiently
def process_csv(file_path):
    frequent_words = Counter()

    # Process CSV in chunks
    for chunk in pd.read_csv(file_path, dtype=str, chunksize=chunk_size):
        chunk.fillna("", inplace=True)  # Fill missing values


        # Tokenize and count words across chunks
        for text in chunk['content']:
            words = word_tokenize(text)
            frequent_words.update(words)

        # Keep only top 10,000 words
        frequent_words = Counter(dict(frequent_words.most_common(10000)))

    return frequent_words

# Process both original and cleaned data
original_words = process_csv(original_file)
cleaned_words = process_csv(cleaned_file)

# Print statistics
print("For original file:")
print(f"100 most frequent words: {original_words.most_common(100)}")

print("\nFor cleaned file:")
print(f"100 most frequent words: {cleaned_words.most_common(100)}")

# Function to plot top words
def plot_top_words(frequent_words, title):
    top_words = frequent_words.most_common(10000)
    words, counts = zip(*top_words) if top_words else ([], [])
    plt.figure(figsize=(20, 10))
    plt.bar(words[:10000], counts[:10000])
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()

# Plot for original file
plot_top_words(original_words, "Top 10,000 Most Frequent Words in Original File")

# Plot for cleaned file
plot_top_words(cleaned_words, "Top 10,000 Most Frequent Words in Cleaned File")

