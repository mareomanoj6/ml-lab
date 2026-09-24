import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('iris.csv')
df['target'] = (df['species'] == 'setosa').astype(int)

X = df[['sepal length (cm)', 'sepal width (cm)']].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

w = clf.coef_[0]
b = clf.intercept_[0]
margin = 2.0 / np.linalg.norm(w)
print(f"Weight vector (w): {w}")
print(f"Bias (b): {b:.4f}")
print(f"Margin (2/||w||): {margin:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, levels=[0], colors='red')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Linear SVM Decision Boundary (Setosa vs Non-Setosa)')
plt.show()

