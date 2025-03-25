import nltk # For text processing
import pandas as pd # For reading/writing CSV files, and working with DataFrame
import os # For generating new output filenames
from nltk.corpus import stopwords # Predefined list of stopwords
from nltk.stem import PorterStemmer # Stemming algorithm to reduce words to root
from nltk.tokenize import word_tokenize # Tokenizer to split text into words

# Ensure necessary downloads, run these lines once if not already downloaded
# nltk.download('stopwords')
# nltk.download('punkt')

# Function to clean and preprocess text data from a CSV file
def cleanF(file_path):
    # Load the CSV file into a DataFrame (treat all values as strings)
    df = pd.read_csv(file_path, dtype=str)
    
    # Ensure stopwords are lowercase English stopwords for filtering
    stop_words = set(w.lower() for w in stopwords.words('english'))
    stemmer = PorterStemmer()

    # Keep track of the vocabulary at each stage
    original_vocab = set() # Before filtering
    filtered_vocab = set() # After removing stopwords and non-alpha 
    stemmed_vocab = set() # After stemming

    # Function to clean a single text string
    def clean_text(text):
        if not isinstance(text, str):
            text = str(text) # Convert non-string values to string

        # Tokenize the text into words via NLTK's tokenizer 
        word_tokens = word_tokenize(text)
        original_vocab.update(word_tokens) # Track original tokens

        # Remove stopwords and non-alphabetic tokens, convert to lowercase
        filtered_tokens = [w.lower() for w in word_tokens if w.isalpha() and w.lower() not in stop_words]
        filtered_vocab.update(filtered_tokens) # Track tokens after stopword removal

        # Use stemmeing to reduce words to their root
        stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
        stemmed_vocab.update(stemmed_tokens) # Track stemmed tokens

        # Reconstruct cleaned text
        return " ".join(stemmed_tokens)

    # Apply cleaning function to 'content' and 'summary' columns if they exist
    for col in ['content', 'summary']:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Compute vocabulary statistics
    OG_vocab_size = len(original_vocab)
    filtered_vocab_size = len(filtered_vocab)
    stemmed_vocab_size = len(stemmed_vocab)

    # Calculate reduction rates after stopword removal and stemming
    stopword_reduction_rate = ((OG_vocab_size - filtered_vocab_size) / OG_vocab_size * 100) if OG_vocab_size else 0
    stemming_reduction_rate = ((filtered_vocab_size - stemmed_vocab_size) / filtered_vocab_size * 100) if filtered_vocab_size else 0

    # Display vocabulary sizes adn reduction percentages
    print(f"Original Vocabulary Size: {OG_vocab_size}")
    print(f"Vocabulary Size After Stopword Removal: {filtered_vocab_size}")
    print(f"Reduction Rate After Stopword Removal: {stopword_reduction_rate:.2f}%")
    print(f"Vocabulary Size After Stemming: {stemmed_vocab_size}")
    print(f"Reduction Rate After Stemming: {stemming_reduction_rate:.2f}%")

    # Create output filename by inserting '_cleaned' before the file extension
    base_name, ext = os.path.splitext(file_path)
    new_file_name = f"{base_name}_cleaned{ext}"
    
    # Save the cleaned DataFrame to a new CSV
    df.to_csv(new_file_name, index=False)
    print(f" Successfully saved cleaned dataset: {new_file_name}")

# List of CSV files to process
file_paths = ['news_sample.csv', '995,000_rows.csv']
for file_path in file_paths:
    cleanF(file_path)
