import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('winequality-red.csv', sep=';')
X = df.drop('quality', axis=1)
y = df['quality']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Experiment with architectures
architectures = [
    (100,),          # 1 hidden layer, 100 neurons
    (100, 50),       # 2 hidden layers, 100 then 50 neurons
    (100, 50, 30)    # 3 hidden layers, 100 then 50 then 30 neurons
]

for arch in architectures:
    mlp = MLPClassifier(hidden_layer_sizes=arch, max_iter=100000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    print(f"Architecture {arch}: Accuracy = {accuracy_score(y_test, y_pred):.4f}")

