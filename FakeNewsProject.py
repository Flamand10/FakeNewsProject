import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# nltk.download('stopwords') for at det virker

#nltk stopwords
stop_words = set(stopwords.words('english'))

#read csv
df = pd.read_csv('exam/news_sample.csv', encoding='utf-8')

def cleanF(text) :
     #makes csv to a single string
    text = " ".join(text.astype(str).agg(" ".join, axis=1))

    #tokenize without punkt, only words
    word_tokens = re.findall(r'\b\w+\b', text)

    #original vocab length
    OG_vocab_size = len(set(word_tokens))

    #remove stopwords
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
# print(cleanF(df))

FakeNewsCorpus = pd.read_csv('exam/995,000_rows.csv', dtype=str)
cleanF(FakeNewsCorpus)