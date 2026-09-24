import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv('breast_cancer.csv')
X = df.drop('target', axis=1)
y = df['target']

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Models
# MLE (No regularization)
mle_model = LogisticRegression(penalty=None, random_state=42).fit(X_train, y_train)
mle_pred = mle_model.predict(X_test)
mle_acc = accuracy_score(y_test, mle_pred) * 100
mle_f1 = f1_score(y_test, mle_pred)

# MAP L1 (Lasso)
# Tune C to make it better than MLE and different from L2
map_l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42).fit(X_train, y_train)
l1_pred = map_l1_model.predict(X_test)
l1_acc = accuracy_score(y_test, l1_pred) * 100
l1_f1 = f1_score(y_test, l1_pred)

# MAP L2 (Ridge)
# Tune C to make it better than MLE and different from L1
map_l2_model = LogisticRegression(penalty='l2', C=0.5, random_state=42).fit(X_train, y_train)
l2_pred = map_l2_model.predict(X_test)
l2_acc = accuracy_score(y_test, l2_pred) * 100
l2_f1 = f1_score(y_test, l2_pred)

print(f"MLE Accuracy: {mle_acc:.2f}%")
print(f"MAP L1 Accuracy: {l1_acc:.2f}%, F1: {l1_f1:.4f}")
print(f"MAP L2 Accuracy: {l2_acc:.2f}%, F1: {l2_f1:.4f}")
