import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

def run_experiment():
    df = pd.read_csv('iris.csv')
    X = df.drop('species', axis=1)
    y = df['species']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=df.drop('species', axis=1).columns)

    # K-Fold Cross Validation
    cv_results = cross_validate(RandomForestClassifier(), X, y, cv=5, scoring=['accuracy', 'f1_macro'])
    print(f"CV Accuracy: {np.mean(cv_results['test_accuracy']):.4f} (+/- {np.std(cv_results['test_accuracy']):.4f})")
    print(f"CV F1 Score: {np.mean(cv_results['test_f1_macro']):.4f} (+/- {np.std(cv_results['test_f1_macro']):.4f})")

    # Bootstrapping
    boot_acc, boot_f1 = [], []
    for _ in range(150):
        idx = np.random.choice(len(X), len(X), replace=True)
        X_b, y_b = X.iloc[idx], y.iloc[idx]
        X_train, X_test, y_train, y_test = train_test_split(X_b, y_b, test_size=0.2)
        model = RandomForestClassifier().fit(X_train, y_train)
        preds = model.predict(X_test)
        boot_acc.append(accuracy_score(y_test, preds))
        boot_f1.append(f1_score(y_test, preds, average='macro'))
    
    print(f"Bootstrapping Accuracy: {np.mean(boot_acc):.4f} (+/- {np.std(boot_acc):.4f})")
    print(f"Bootstrapping F1 Score: {np.mean(boot_f1):.4f} (+/- {np.std(boot_f1):.4f})")

if __name__ == "__main__":
    run_experiment()
