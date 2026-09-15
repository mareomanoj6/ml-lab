import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('fashion-mnist.csv')

X = df.drop('label', axis=1).values / 255.0
y = df['label'].values

X_subset, _, y_subset, _ = train_test_split(X, y, train_size=2000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

kernels = ['linear', 'poly', 'rbf']
results = {}

for kernel in kernels:
    print(f"Training SVM with {kernel} kernel...")
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[kernel] = acc
    print(f"{kernel} accuracy: {acc:.2f}")

print("\nComparison Summary:")
for k, v in results.items():
    print(f"{k}: {v:.2f}")
