#the code to merge the training set from the dataplit and the 800 articles we scraped in the beginning.
import pandas as pd

# load the two datasets
train = pd.read_csv('train_set.csv', low_memory=False)
articles = pd.read_csv('bbc_articles.csv', low_memory=False)

# check the column names for the two datasets
print("Columns in train_set:", train.columns)
print("Columns in bbc_articles:", articles.columns)

# create the binary labels 0 and 1 in train_set.csv
train['label'] = train['type'].apply(lambda x: 1 if x == 'reliable' else 0)

# assign the labels to the BBC articles
articles['label'] = 1

# change the name 'summary' to 'content' in bbc_articles.csv
articles.rename(columns={'summary': 'content'}, inplace=True)

# concatenate DataFrames (include all relevant columns)
joined = pd.concat([train[['content', 'label']], articles[['content', 'label']]], ignore_index=True)

# save the merged data to the joint_contents.csv
try:
    joined.to_csv('joint_contents.csv', index=False)
    print("The merged content has been successfully saved to 'joint_contents.csv'.")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")
