# FakeNewsProject

[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\Caspe\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package punkt to
[nltk_data]     C:\Users\Caspe\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt is already up-to-date!
Original Vocabulary Size: 22704
Vocabulary Size After Stopword Removal: 15415
Reduction Rate After Stopword Removal: 32.10%
Vocabulary Size After Stemming: 10226
Reduction Rate After Stemming: 33.66%
 Successfully saved cleaned dataset: news_sample_cleaned.csv
Original Vocabulary Size: 5637631
Vocabulary Size After Stopword Removal: 1259204
Reduction Rate After Stopword Removal: 77.66%
Vocabulary Size After Stemming: 1098642
Reduction Rate After Stemming: 12.75%
 Successfully saved cleaned dataset: 995,000_rows_cleaned.csv




### Result from '800AndLiarClean.ipynb':
- "Processing columns: ['region', 'title', 'summary', 'link'] in bbc_articles.csv <br>

   Stats for bbc_articles.csv:
   Original Vocabulary Size: 6732
   Vocabulary Size After Stopword Removal: 4849
   Reduction Rate After Stopword Removal: 27.97%
   Vocabulary Size After Stemming: 3707
   Reduction Rate After Stemming: 23.55%
   Successfully saved cleaned dataset: bbc_articles_cleaned.csv
   Processing columns: ['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_13'] in test.tsv <br>
   
   Stats for test.tsv:
   Original Vocabulary Size: 7464
   Vocabulary Size After Stopword Removal: 4339
   Reduction Rate After Stopword Removal: 41.87%
   Vocabulary Size After Stemming: 3303
   Reduction Rate After Stemming: 23.88%
   Successfully saved cleaned dataset: test_cleaned.tsv" <br><br>





# Part 4

## Result from 'Part4Evaluation.ipynb':
This notebook evaluates the performance of an SVM + TF-IDF model on three different test sets:  
1. The original FakeNewsCorpus  
2. The FakeNewsCorpus extended with 800 scraped articles  
3. The LIAR dataset (cross-domain evaluation)
   
---

### Results Summary

| Dataset                     | Validation F1 | Test F1 |
|----------------------------|---------------|---------|
| FakeNewsCorpus             | 0.89          | 0.89    |
| FNC + 800 scraped articles | 0.89          | 0.90    |
| LIAR (cross-domain)        | 0.89          | 0.03    |

> Note: Low F1 on LIAR is expected due to class imbalance and domain shif

---

### Required Files

Place the following files in the **same directory** as `Part4Evaluation.ipynb`:

- `train_set.csv`  
- `val_set.csv`  
- **One of the following** test sets depending on what you're running:
  - `test_set.csv` — original FakeNewsCorpus test split
  - `joint_contents.csv` — FakeNewsCorpus + scraped articles
  - `test_cleaned.tsv` — LIAR dataset (saved as `.tsv` but is comma-separated)

---

### How to Run the Notebook

1. Make sure Python is installed and activate your environment.
2. Install required dependencies:

```bash
pip install pandas scikit-learn
