import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame

df['target'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})
df.rename(columns={'target': 'species'}, inplace=True)

df.to_csv('iris.csv', index=False)
print("CSV generated successfully!")

