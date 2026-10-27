from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score 
import pandas as pd 

df = pd.read_csv("Datasets/fashion_mnist.csv")
df = df.sample(5000, random_state=42) # take 5000 samples (optimization)
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

kernels = ["linear", "poly", "rbf"]
acs = [] ; pcs = [] ; rcs = []; f1s = []
for k in kernels:
  if k == 'poly':
    model = SVC(kernel="poly", degree=3, max_iter=1000, random_state=42)
  else:
    model = SVC(kernel=k, max_iter=1000, random_state=42)

  model.fit(X_train, y_train)
  pred = model.predict(X_test)

  acs.append(accuracy_score(y_test, pred))
  pcs.append(precision_score(y_test, pred, average='weighted'))
  rcs.append(recall_score(y_test, pred, average='weighted'))
  f1s.append(f1_score(y_test, pred, average='weighted'))
results = pd.DataFrame({
    "Kernel": kernels,
    "Accuracy":acs,
    "Precision":pcs,
    "Recall":rcs,
    "F1 Score":f1s
})
numeric = ["Accuracy", "Precision", "Recall", "F1 Score"]
results[numeric] = results[numeric].map(lambda x: f"{x:.4f}")
print(results.to_string(index=False))
