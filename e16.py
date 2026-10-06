import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def run_experiment():
    df = pd.read_csv('iris.csv')
    X = df.drop('species', axis=1)
    y = df['species']

    # K-Fold Cross Validation
    cv_scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)
    print(f"CV Accuracy: {np.mean(cv_scores):.2f}")

    # Bootstrapping
    boot_scores = []
    for _ in range(100):
        idx = np.random.choice(len(X), len(X), replace=True)
        X_b, y_b = X.iloc[idx], y.iloc[idx]
        # Use a simple split for evaluation within bootstrap
        X_train, X_test, y_train, y_test = train_test_split(X_b, y_b, test_size=0.3)
        model = RandomForestClassifier().fit(X_train, y_train)
        boot_scores.append(accuracy_score(y_test, model.predict(X_test)))
    
    print(f"Bootstrapping Accuracy: {np.mean(boot_scores):.2f}")

if __name__ == "__main__":
    run_experiment()
