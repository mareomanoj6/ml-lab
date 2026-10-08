import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_experiment():
    df = pd.read_parquet('mnist.parquet')
    X = df.drop('class', axis=1) / 255.0
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    for act in ['logistic', 'relu', 'tanh']:
        mlp = MLPClassifier(activation=act, max_iter=5).fit(X_train, y_train)
        print(f"Act {act} Accuracy: {accuracy_score(y_test, mlp.predict(X_test)):.4f}")

if __name__ == "__main__":
    run_experiment()
