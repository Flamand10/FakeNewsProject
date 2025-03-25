import pandas as pd # For reading CSV and handeling data in chunks
from nltk.tokenize import word_tokenize # Tokenizer to split text into words
from collections import Counter # For counting word frequencies 
import matplotlib.pyplot as plt # For plotting word frequency in bar charts
import re


# File paths for original and cleaned datasets
original_file = "995,000_rows.csv"
cleaned_file = "995,000_rows_cleaned.csv"

# Chunk size for reading large csv in memory effecient way
chunk_size = 10000  

# Function to process a CSV file and count word frequencies
def process_csv(file_path):
    frequent_words = Counter()

    # Read CSV in chunks to handle large files 
    for chunk in pd.read_csv(file_path, dtype=str, chunksize=chunk_size):
        chunk.fillna("", inplace=True)  # Replace NaNs with empty strings

        # Tokenize and count words in the 'content' column
        for text in chunk['content']:
            words = word_tokenize(text)
            frequent_words.update(words)

        # Keep only top 10,000 most frequent words to memory grow indefinitely
        frequent_words = Counter(dict(frequent_words.most_common(10000)))

    return frequent_words

# Count word frequencies in both original and cleaned datasets
original_words = process_csv(original_file)
cleaned_words = process_csv(cleaned_file)

# Print top 100 most frequent words in each file
print("For original file:")
print(f"100 most frequent words: {original_words.most_common(100)}")

print("\nFor cleaned file:")
print(f"100 most frequent words: {cleaned_words.most_common(100)}")

# Function to plot top words using bar chart
def plot_top_words(frequent_words, title):
    top_words = frequent_words.most_common(10000)
    words, counts = zip(*top_words) if top_words else ([], [])
    plt.figure(figsize=(20, 10))
    plt.bar(words[:10000], counts[:10000]) # Plot up to 10000 words
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()

# Plot word frequencies for original dataset 
plot_top_words(original_words, "Top 10,000 Most Frequent Words in Original File")

# Plot word freqeucies for cleaned dataset
plot_top_words(cleaned_words, "Top 10,000 Most Frequent Words in Cleaned File")

