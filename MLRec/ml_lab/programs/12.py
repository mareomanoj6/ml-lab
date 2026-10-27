from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
import pandas as pd 

df = pd.read_csv("Datasets/wine_quality.csv")
X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

archs = [(10,), (100,), (20,10), (50, 100, 30)]
accs = [] ; precs = [] ; recs = [] ; f1s = []
for arch in archs:
  model = MLPClassifier(hidden_layer_sizes=arch, max_iter=2000, early_stopping=True, random_state=42)
  model.fit(X_train, y_train)
  pred = model.predict(X_test)

  accs.append(accuracy_score(y_test, pred))
  precs.append(precision_score(y_test, pred, average='weighted', zero_division=0))
  recs.append(recall_score(y_test, pred, average='weighted'))
  f1s.append(f1_score(y_test, pred, average='weighted'))
results = pd.DataFrame({
    "Architecture": [str(a) for a in archs],
    "Accuracy": accs,
    "Precision": precs,
    "Recall": recs,
    "F1 Score": f1s,
})
numeric = ["Accuracy", "Precision", "Recall", "F1 Score"]
results[numeric] = results[numeric].map(lambda x: f"{x:.4f}")
print(results.to_string(index=False))