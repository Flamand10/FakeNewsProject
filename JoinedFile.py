import pandas as pd

train = pd.read_csv('train_set.csv', dtype=str)
articles = pd.read_csv('XXXXXX.csv', dtype=str)

# concat content columns from both files.
join = pd.concat([train['content'], articles['content']], ignore_index=True)

#joining the data.
joined = pd.DataFrame({'content': join})

joined.to_csv('joined_contents.csv', index=False)