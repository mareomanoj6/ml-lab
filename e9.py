import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('iris.csv')
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, impurity=False, node_ids=False)
for text in plt.gca().texts:
    lines = text.get_text().split('\n')
    filtered = [l for l in lines if not any(k in l for k in ['samples', 'value', 'impurity'])]
    if filtered:
        text.set_text(filtered[0].replace('class = ', '').strip())
plt.title("Decision Tree (ID3 Logic - Entropy)")
plt.show()

importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:\n", importance)

