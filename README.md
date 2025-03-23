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







### Result from 'Part4Evaluation.ipynb':
- Evaluated on fake news corpus:
 - "Columns in train_set: Index(['Unnamed: 0', 'id', 'domain', 'type', 'url', 'content', 'scraped_at',
       'inserted_at', 'updated_at', 'title', 'authors', 'keywords',
       'meta_keywords', 'meta_description', 'tags', 'summary', 'source'],
      dtype='object')
Columns in val_set: Index(['Unnamed: 0', 'id', 'domain', 'type', 'url', 'content', 'scraped_at',
       'inserted_at', 'updated_at', 'title', 'authors', 'keywords',
       'meta_keywords', 'meta_description', 'tags', 'summary', 'source'],
      dtype='object')
Columns in test_set: Index(['Unnamed: 0', 'id', 'domain', 'type', 'url', 'content', 'scraped_at',
       'inserted_at', 'updated_at', 'title', 'authors', 'keywords',
       'meta_keywords', 'meta_description', 'tags', 'summary', 'source'],
      dtype='object')
F1 Score on Validation Set (SVM + TF-IDF): 0.89
F1 Score on Test Set (SVM + TF-IDF): 0.89" <br>


- Evaluated on fake news corpus + 800 scraped articles:
 - "Columns in train_set: Index(['Unnamed: 0', 'id', 'domain', 'type', 'url', 'content', 'scraped_at',
       'inserted_at', 'updated_at', 'title', 'authors', 'keywords',
       'meta_keywords', 'meta_description', 'tags', 'summary', 'source'],
      dtype='object')
Columns in val_set: Index(['Unnamed: 0', 'id', 'domain', 'type', 'url', 'content', 'scraped_at',
       'inserted_at', 'updated_at', 'title', 'authors', 'keywords',
       'meta_keywords', 'meta_description', 'tags', 'summary', 'source'],
      dtype='object')
Columns in test_set: Index(['content', 'label'], dtype='object')
F1 Score on Validation Set (SVM + TF-IDF): 0.89
F1 Score on Test Set (SVM + TF-IDF): 0.90" <br>

- Evaluation on LIAR dataset:
 - "Columns in train_set: Index(['Unnamed: 0', 'id', 'domain', 'type', 'url', 'content', 'scraped_at',
       'inserted_at', 'updated_at', 'title', 'authors', 'keywords',
       'meta_keywords', 'meta_description', 'tags', 'summary', 'source'],
      dtype='object')
Columns in val_set: Index(['Unnamed: 0', 'id', 'domain', 'type', 'url', 'content', 'scraped_at',
       'inserted_at', 'updated_at', 'title', 'authors', 'keywords',
       'meta_keywords', 'meta_description', 'tags', 'summary', 'source'],
      dtype='object')
Columns in test_set: Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 'label', 'content'], dtype='object')
F1 Score on Validation Set (SVM + TF-IDF): 0.89
F1 Score on Test Set (SVM + TF-IDF): 0.03" 
