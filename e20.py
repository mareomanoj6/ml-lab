import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_experiment():
    df = pd.read_parquet('fashion-mnist.parquet')
    X = df.drop('label', axis=1) / 255.0
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = [
        {'lr': 0.001, 'batch': 32, 'epochs': 10},
        {'lr': 0.01, 'batch': 64, 'epochs': 10},
    ]

    for p in params:
        mlp = MLPClassifier(learning_rate_init=p['lr'], batch_size=p['batch'], max_iter=p['epochs']).fit(X_train, y_train)
        print(f"LR={p['lr']}, Batch={p['batch']} Acc: {accuracy_score(y_test, mlp.predict(X_test)):.2f}")

if __name__ == "__main__":
    run_experiment()
