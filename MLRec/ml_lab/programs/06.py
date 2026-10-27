from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Datasets/breastcancer.csv")
X = df.drop(columns=["target"])
y = df[["target"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLE 
mle = LogisticRegression(penalty=None, max_iter=1000)
mle.fit(X_train, y_train)
y_pred_mle = mle.predict(X_test)

# MAP Using L1
map_l1 = LogisticRegression(penalty="l1", solver='liblinear', C=1.0, max_iter=1000)
map_l1.fit(X_train, y_train)
y_pred_l1 = map_l1.predict(X_test)

# MAP using L2
map_l2 = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
map_l2.fit(X_train, y_train)
y_pred_l2 = map_l2.predict(X_test)

print("Logistic Regression MLE: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_mle): .4f}\nPrecision: {precision_score(y_test, y_pred_mle): .4f} \nRecall: {recall_score(y_test, y_pred_mle): .4f} \nF1 Score: {f1_score(y_test, y_pred_mle): .4f}\n")

print("Logistic Regression MAP (L1): ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_l1): .4f}\nPrecision: {precision_score(y_test, y_pred_l1): .4f} \nRecall: {recall_score(y_test, y_pred_l1): .4f} \nF1 Score: {f1_score(y_test, y_pred_l1): .4f}\n")

print("Logistic Regression MAP (L2): ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_l2): .4f}\nPrecision: {precision_score(y_test, y_pred_l2): .4f} \nRecall: {recall_score(y_test, y_pred_l2): .4f} \nF1 Score: {f1_score(y_test, y_pred_l2): .4f}\n")