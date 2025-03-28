# All libraries needed to run the code
- nltk
- pandas as pd
- os
- re
- collections
- matplotlib.pyplot
- nltk.download('stopwords')
- nltk.download('punkt')
- sklearn
- wandb



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
## Result from `Counting_Analysis.py`

This script analyzes the most frequent words in the original and cleaned datasets by:
- Tokenizing the text content using NLTK
- Counting word frequency with `collections.Counter`
- Plotting the top 100 most frequent words from each file
- Logging both the frequency data and plots to Weights & Biases (wandb)

This allows for a direct comparison of how cleaning affects the vocabulary distribution.

This script was run on two CSV files:  
1. `995,000_rows.csv`  
2. `995,000_rows_cleaned.csv`  

### Output Summary

- Logged the 100 most frequent words (before and after cleaning) to wandb
- Generated frequency bar plots for both versions of the dataset
- Demonstrated how stopword removal and stemming significantly reduce noisy or common word presence

> Link to Wandb: https://wandb.ai/zxt667-university-of-copenhagen/text-cleaning-vocab-analysis/runs/6caa010y/workspace

---
Below are the word frequency distributions from the original and cleaned datasets. These help visualize the impact of preprocessing steps such as stopword removal and stemming.

### Word Frequency Visualization

#### Original Dataset — Full Vocabulary Distribution
Original 100 most frequent words:
![Original Plot 1](media_images_original_freq_plot_1_0e4f988ea9b64a0b9548(1).png)  

Original 10000 most frequent words:
![Original Plot 2](media_images_original_freq_plot_1_9ae4c383e178bc54f0f8(1).png)

#### Cleaned Dataset — Vocabulary After Preprocessing
Cleaned 100 most frequent words:
![Cleaned Plot 1](media_images_cleaned_freq_plot_2_89bb85b7a66c1e248cc1.png)  

Cleaned 10000 most frequent words:
![Cleaned Plot 2](media_images_cleaned_freq_plot_2_d46bd1d5bb55c07e23cd.png).


---

### Required Files

Place the following input CSVs in the **same directory** as `Counting_Analysis.py`:

- `995,000_rows.csv`  
- `995,000_rows_cleaned.csv`  

Each file must include a `content` column with the text to analyze.

---

### How to Run the Script

1. Make sure Python is installed and your environment is active.
2. Install the required dependencies:
'pip install pandas nltk matplotlib wandb'
3. Follow the wandb instructions in terminal, to plot to your user


---
## Result from the `SplittingData.ipynb`

This script loads a cleaned dataset (`995,000_rows_cleaned.csv`) and splits it into:
- **Training set (80%)**
- **Validation set (10%)**
- **Test set (10%)**

Each split is saved as a separate CSV file for use in later stages of the pipeline.

---

The script was run on `995,000_rows_cleaned.csv` and produced the following splits:

### Split Summary

| Split         | Rows     |
|---------------|----------|
| Training Set  | 796,000  |
| Validation Set| 99,500   |
| Test Set      | 99,500   |

> Split files were saved as:
> - `train_set.csv`  
> - `val_set.csv`  
> - `test_set.csv`

---

### Required File

Place the following input file in the **same directory** as the script:

- `995,000_rows_cleaned.csv` — fully preprocessed dataset

---

### How to Run the Script

1. Make sure Python is installed and activate your environment.
2. Install the required dependency:
'pip install pandas scikit-learn'

### Exploring data
In the file `Display_af_995000.ipynb`We show our representation of the data set for manual exploration.
The file required to run the sscript is `995,000_rows_cleaned.csv`.

<br><br>

# Part 2: Simple Logistic Regression Model

## Cleaning the articles from graded assignment 2

### Result from `800AndLiarClean.ipynb`

This notebook was run on two datasets:  
1. `bbc_articles.csv` (scraped articles)  
2. `test.tsv` (LIAR dataset)  (Will be used for part 4 later, but is just in the same notebook as the cleaning of the articles)

### Vocabulary Reduction Summary

| Dataset           | Original Vocab | After Stopword Removal | Reduction (%) | After Stemming | Reduction (%) |
|-------------------|----------------|-------------------------|----------------|----------------|----------------|
| bbc_articles.csv  | 6,732          | 4,849                   | 27.97%         | 3,707          | 23.55%         |
| test.tsv          | 7,464          | 4,339                   | 41.87%         | 3,303          | 23.88%         |

> Cleaned files were saved as:
> - `bbc_articles_cleaned.csv`  
> - `test_cleaned.tsv` (Will be used in part 4)

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
# Logistic Regression Analysis

This Jupyter Notebook implements a **Logistic Regression model** to classify news articles as either **reliable** or **fake**. It follows previous preprocessing steps and builds a baseline model using simple text vectorization and logistic regression. 

---

## **Workflow Overview**
### **1. Load the Dataset**
- Reads the preprocessed datasets:
  - `train_set.csv`
  - `val_set.csv`
  - `test_set.csv`
- Verifies the presence of required columns.

### **2. Preprocessing and Labeling**
- Converts article labels into **binary labels**:
  - **1** → "reliable"
  - **0** → "fake"

### **3. Feature Extraction**
- Uses **CountVectorizer** (Bag of Words model) to transform text into numerical features.

### **4. Train a Logistic Regression Model**
- Fits a **Logistic Regression classifier** on the training data.
- Uses **default hyperparameters** to establish a baseline model.

### **5. Model Evaluation**
- Evaluates the model using:
  - **Validation F1 Score**
  - **Test F1 Score**
- It then prints the f1 scores for the validation set and test set. 

---
# **Result from the logistic_regression.ipynb**

## **Required Files**
Make sure the following files are in the **same directory** as the notebook:
- `train_set.csv`  
- `val_set.csv`  
- `test_set.csv`  

---

## **How to Run the Notebook**
1. Install dependencies:
   1. Pandas
   2. Scikit-learn
      1. CountVectorizer
      2. LogisticRegression
      3. f1_score
2. Make sure that you have the required datasets in the same directory
   1. The train set
   2. The validation set
   3. The test set
3. 
## **How this code is run**
This code in this notebook is run 3 times, firstly it is run on the train, validation and test set produced from the SplittingData.ipynb. For which we need the: 
1. Train set (train_set.csv)
2. Validation set (val_set.csv)
3. Test set (test_set.csv)

Secondly the code is run for the joint.ipynb file where the data is trained on the train_set.csv that is then merged with the original extra reliable 800 articles we scraped from BBC. 

Lastly the code is run again trained on the original train_set.csv, where the test and validation set are from the LIAR dataset. These datasets are then cleaned and run later in part 4 (test_cleaned.tsv). 
#### Result from the logistic regression analysis on the 3 datasets
| Dataset                     | Validation F1 | Test F1 |
|----------------------------|---------------|---------|
| train_csv             | 0.86          | 0.86    |
| Joint.csv | 0.86          | 0.86    |
| test_cleaned.tsv        | 0.04          | 0.01    |

>Note: that the test_cleaned.tsv dataset is not run in part two but instead later in part 4, as it is the liar dataset.
# **Result from JoinedFile.py**

This script merges two datasets:
1. The original training set (`train_set.csv`)
2. Scraped BBC articles (`bbc_articles.csv`)

The merged dataset is saved as `joint_contents.csv` with consistent labeling.

### Prerequisites

- Python 3.x
- pandas library
# Part 3: Advanced Model
## Code for the advanced model
Look at the code in the jupiter notebook file `Part4Evaluation.ipynb` In the file we run the code for part 4. No code had to be ran in part 3.

<br><br>

# Part 4: Evaluation

## Requirements
Make sure to do all previous steps for running the code for all parts up until part 4. Have the required file in the ssame directory as the code file.

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
- `test_set.csv` — original FakeNewsCorpus test split  
- `joint_contents.csv` — FakeNewsCorpus + scraped articles  
- `test_cleaned.tsv` — LIAR dataset (saved as `.tsv` but is comma-separated)

---

### How to Run the Notebook

1. Make sure Python is installed and activate your environment.
2. Install required dependencies:
'pip install pandas scikit-learn'
