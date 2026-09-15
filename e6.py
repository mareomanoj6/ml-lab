import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("breast_cancer.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nMLE")

mle = LogisticRegression(
        penalty=None,
        max_iter=5000
    )

mle.fit(X_train, y_train)

y_pred = mle.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nMAP - L2")

map_l2 = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=5000
)

map_l2.fit(X_train, y_train)

y_pred = map_l2.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nMAP - L1")

map_l1 = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1.0,
    max_iter=5000
)

map_l1.fit(X_train, y_train)

y_pred = map_l1.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

mle_coef = mle.coef_[0]
l2_coef = map_l2.coef_[0]
l1_coef = map_l1.coef_[0]

print("\nMLE Parameters:")
print(mle_coef)

print("\nMAP L2 Parameters:")
print(l2_coef)

print("\nMAP L1 Parameters:")
print(l1_coef)

