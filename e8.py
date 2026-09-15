import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

def run_experiment():
    df = pd.read_csv('fashion-mnist.csv')
    
    X = df.drop('label', axis=1).values / 255.0
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, test_size=1000, random_state=42, stratify=y)
    
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    results = {}
    
    print(f"Evaluating KNN on Fashion MNIST (Train size: {len(X_train)}, Test size: {len(X_test)})")
    print("-" * 50)
    
    for k in k_values:
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        preds = knn.predict(X_test)
        acc = accuracy_score(y_test, preds)
        duration = time.time() - start_time
        
        results[k] = (acc, duration)
        print(f"K={k} | Accuracy: {acc:.4f} | Time: {duration:.2f}s")
    
    print("-" * 50)
    best_k = max(results, key=lambda k: results[k][0])
    print(f"Best K: {best_k} with Accuracy: {results[best_k][0]:.4f}")

if __name__ == "__main__":
    run_experiment()
