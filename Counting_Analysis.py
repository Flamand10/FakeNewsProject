import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import re
import tldextract
import dateparser

df1 = pd.read_csv('995,000_rows.csv', dtype=str)
df1 = df1.fillna("")
df2 = pd.read_csv('995,000_rows_cleaned.csv', dtype=str)
df2 = df1.fillna("")

# Uses tldextract to count number of urls in content.
def url(text):
    words = text.split()
    return sum(1 for word in words if tldextract.extract(word).suffix)

# Uses dateparser to count the dates in content.
def dates(text):
    words = text.split()
    return sum(1 for word in words if dateparser.parse(word) is not None)

# Uses re to count rumber of numeric values in content.
def numeric(text):
    return len(re.findall(r'\b\d+\b', text))

# Uses counter to find frequent 100 words in content.
def frequent100(text):
    words = word_tokenize(text)
    wordCount = Counter(words)
    return wordCount.most_common(100)

# Uses counter to find frequent 10000 words in content.
def frequent10000(text):
    words = word_tokenize(text)
    wordCount = Counter(words)
    return wordCount.most_common(10000)

# Printing for original data:
for column, df in [('content', df1)]:
    print("For original file:")
    print(f"Url's: {df1['content'].apply(url).sum()}")
    print(f"Dates: {df1['content'].apply(dates).sum()}")
    print(f"Numeric values: {df1['content'].apply(numeric).sum()}")
    print(f"100 most frequent words: {df1['content'].apply(lambda x: frequent100(' '.join(df1['content'])))}")

# printing for Cleaned data:
for column, df in [('content', df2)]:
    print("For cleaned file:")
    print(f"Url's: {df2['content'].apply(url).sum()}")
    print(f"Dates: {df2['content'].apply(dates).sum()}")
    print(f"Numeric values: {df2['content'].apply(numeric).sum()}")
    print(f"100 most frequent words: {df2['content'].apply(lambda x: frequent100(' '.join(df2['content'])))}")

# Plot the 10000 most frequent words for the original data.
original_text = ' '.join(df1['content'])
original_words = frequent10000(original_text)
words, counts = zip(*original_words)
plt.figure(figsize=(20, 10))
plt.bar(words[:10000], counts[:10000])
plt.xticks(rotation=90)
plt.title('Top 10000 Most Frequent Words in Original File')
plt.show()

# Plot the 10000 most frequent words for the cleaned data:
cleaned_text = ' '.join(df2['content'])
cleaned_words = frequent10000(cleaned_text)
words, counts = zip(*cleaned_words)
plt.figure(figsize=(20, 10))
plt.bar(words[:10000], counts[:10000])
plt.xticks(rotation=90)
plt.title('Top 10000 Most Frequent Words in Cleaned File')
plt.show()