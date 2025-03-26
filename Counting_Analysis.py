import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import wandb

# Initialize wandb project
wandb.init(project="text-cleaning-vocab-analysis", name="vocab_stats_and_plots")

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

    # Keep only top 10,000 words (safety cap)
    return Counter(dict(frequent_words.most_common(10000)))

# Process both original and cleaned data
original_words = process_csv(original_file)
cleaned_words = process_csv(cleaned_file)

# Log top 10,000 frequent words to wandb as tables
original_table = wandb.Table(columns=["word", "count"], data=original_words.most_common(10000))
cleaned_table = wandb.Table(columns=["word", "count"], data=cleaned_words.most_common(10000))

wandb.log({
    "original_top_10000_table": original_table,
    "cleaned_top_10000_table": cleaned_table
})

# Function to plot top 10,000 words in a readable way
def plot_top_words(frequent_words, title, tag):
    top_words = frequent_words.most_common(10000)
    if not top_words:
        return

    words, counts = zip(*top_words)
    plt.figure(figsize=(60, 10))  # Wider figure for more space
    plt.bar(range(len(words)), counts, width=1.0)  # Index-based bars
    plt.title(title)
    plt.xlabel("Words (Index)")
    plt.ylabel("Frequency")

    # Optionally annotate the top 1000 bars with word labels
    top_n_to_annotate = 1000
    for i in range(top_n_to_annotate):
        plt.text(i, counts[i], words[i], rotation=90, ha='center', va='bottom', fontsize=4)

    plt.xticks([])  # Hide all x-axis tick labels to prevent clutter
    plt.tight_layout()

    # Save plot to wandb
    wandb.log({tag: wandb.Image(plt)})
    plt.close()

# Plot and log for original file
plot_top_words(original_words, "Top 10000 Most Frequent Words in Original File", "original_freq_plot")

# Plot and log for cleaned file
plot_top_words(cleaned_words, "Top 10000 Most Frequent Words in Cleaned File", "cleaned_freq_plot")

# Finish wandb logging
wandb.finish()
