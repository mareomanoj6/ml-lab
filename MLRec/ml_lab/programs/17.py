from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd 

df = pd.read_csv("Datasets/titanic.csv")

# Only useful columns
df = df[["pclass","sex","age","sibsp","parch","fare","embarked","survived"]]

# Filing in null values
num_cols = ["age", "fare"]
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_cols = ["sex", "embarked"]
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

encoder = LabelEncoder()
df["sex"] = encoder.fit_transform(df["sex"])
df["embarked"] = encoder.fit_transform(df["embarked"])

X = df.drop(columns=["survived"])
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

acs, pcs, rcs, f1s = [], [], [], []
models = {
    "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42),
    "Boosting": AdaBoostClassifier(n_estimators=100, random_state=42)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acs.append(accuracy_score(y_test, pred))
    pcs.append(precision_score(y_test, pred))
    rcs.append(recall_score(y_test, pred))
    f1s.append(f1_score(y_test, pred))
results = pd.DataFrame({
    "Method": list(models.keys()),
    "Accuracy": acs,
    "Precision": pcs,
    "Recall": rcs,
    "F1 Score": f1s
})
numeric = ["Accuracy", "Precision", "Recall", "F1 Score"]
results[numeric] = results[numeric].map(lambda x: f"{x:.4f}")
print(results.to_string(index=False))
