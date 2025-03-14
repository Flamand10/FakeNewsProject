import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
# nltk.download('stopwords') for at det virker


#read csv

def cleanF(file_path) :
    df = pd.read_csv(file_path, dtype=str)

    #makes csv to a single string
    text = " ".join(df.astype(str).agg(" ".join, axis=1))

    #tokenize without punkt, only words
    word_tokens = re.findall(r'\b\w+\b', text)

    #original vocab length
    OG_vocab_size = len(set(word_tokens))

    #remove stopwords
    #nltk stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in word_tokens if w.lower() not in stop_words]
    filtered_vocab_size = len(set(filtered_tokens))

    #stopword reduction rate
    stopword_reduction_rate = ((OG_vocab_size - filtered_vocab_size) / OG_vocab_size * 100)

    #stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    stemmed_vocab_size = len(set(stemmed_tokens))

    #stem reduction rate
    stemming_reduction_rate = ((filtered_vocab_size - stemmed_vocab_size) / filtered_vocab_size * 100)

    #print stuff
    print(f"Original Vocabulary Size: {OG_vocab_size}")
    print(f"Vocabulary Size After Stopword Removal: {filtered_vocab_size}")
    print(f"Reduction Rate After Stopword Removal: {stopword_reduction_rate:.2f}%")
    print(f"Vocabulary Size After Stemming: {stemmed_vocab_size}")
    print(f"Reduction Rate After Stemming: {stemming_reduction_rate:.2f}%")

    # Save cleaned text as CSV
    cleaned_text = " ".join(stemmed_tokens)
    
    # Generate new file name with 'cleaned' added to the current CSV name
    base_name, ext = os.path.splitext(file_path)
    new_file_name = f"{base_name}_cleaned{ext}"
    
    cleaned_df = pd.DataFrame([cleaned_text], columns=['cleaned_text'])
    cleaned_df.to_csv(new_file_name, index=False)

# Example usage with multiple files
file_paths = ['news_sample.csv', '995,000_rows.csv']
for file_path in file_paths:
    cleanF(file_path)