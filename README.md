# Part 1: Data Processing 

## Result from `FakeNewsProject.py`

This script cleans and preprocesses raw text data by:
- Tokenizing text into words
- Removing stopwords and non-alphabetic tokens
- Applying stemming
- Saving the cleaned dataset with `_cleaned` added to the filename
- Reporting vocabulary reduction statistics


This script was run on two CSV files:  
1. `news_sample.csv`  
2. `995,000_rows.csv`  

### Vocabulary Reduction Summary

| Dataset              | Original Vocab | After Stopword Removal | Reduction (%) | After Stemming | Reduction (%) |
|----------------------|----------------|-------------------------|----------------|----------------|----------------|
| news_sample.csv      | 22,704         | 15,415                  | 32.10%         | 10,226         | 33.66%         |
| 995,000_rows.csv     | 5,637,631      | 1,259,204               | 77.66%         | 1,098,642      | 12.75%         |

> Cleaned files were saved as:
> - `news_sample_cleaned.csv`  
> - `995,000_rows_cleaned.csv`

---

### Required Files

Place the following input CSVs in the **same directory** as `FakeNewsProject.py`:

- `news_sample.csv`  
- `995,000_rows.csv`  

Each file should contain a `content` column and optionally a `summary` column.

---

### How to Run the Script

1. Make sure Python is installed and activate your environment.
2. Install the required dependencies:
'pip install pandas nltk'
---






# Part 2: Simple Logistic Regression Model





# Part 3: Advanced Model




# Part 4: Evaluation

## Code for the advanced model
Look at the code in the jupiter notebook file `Part4Evaluation.ipynb` In the file we run the code for part 4. No code had to be ran in part 3.



## Result from `800AndLiarClean.ipynb`

This notebook was run on two datasets:  
1. `bbc_articles.csv` (scraped articles)  
2. `test.tsv` (LIAR dataset)

### Vocabulary Reduction Summary

| Dataset           | Original Vocab | After Stopword Removal | Reduction (%) | After Stemming | Reduction (%) |
|-------------------|----------------|-------------------------|----------------|----------------|----------------|
| bbc_articles.csv  | 6,732          | 4,849                   | 27.97%         | 3,707          | 23.55%         |
| test.tsv          | 7,464          | 4,339                   | 41.87%         | 3,303          | 23.88%         |

> Cleaned files were saved as:
> - `bbc_articles_cleaned.csv`  
> - `test_cleaned.tsv`

---

### Required Files

Place the following input files in the **same directory** as `800AndLiarClean.ipynb`:

- `bbc_articles.csv` — scraped article dataset  
- `test.tsv` — original LIAR dataset (TSV file, comma-separated)

---

### How to Run the Notebook

1. Make sure Python is installed and activate your environment.
2. Install the required dependencies:
'pip install pandas nltk'

---

## Result from `Part4Evaluation.ipynb`:
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
| LIAR (cross-domain)        | 0.05          | 0.03    |

> Note: Low F1 on LIAR is expected due to class imbalance and domain shif

---

### Required Files

Place the following files in the **same directory** as `Part4Evaluation.ipynb`:

- `train_set.csv`  
- `val_set.csv` — validation split for the FakeNewsCorpus  
- `valid.tsv` — validation set for the LIAR dataset (tab-separated)  

**One of the following** test sets depending on which dataset you're using:
- `test_set.csv` — original FakeNewsCorpus test split  
- `joint_contents.csv` — FakeNewsCorpus + scraped articles  
- `test_cleaned.tsv` — LIAR dataset (saved as `.tsv` but is comma-separated)

---

### How to Run the Notebook

1. Make sure Python is installed and activate your environment.
2. Install required dependencies:
'pip install pandas scikit-learn'
