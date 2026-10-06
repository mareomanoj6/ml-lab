import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def run_experiment():
    df = pd.read_csv('titanic.csv')
    # Simple preprocessing
    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].dropna()
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Bagging
    bag = BaggingClassifier(estimator=DecisionTreeClassifier()).fit(X_train, y_train)
    # Boosting
    boost = AdaBoostClassifier().fit(X_train, y_train)

    print(f"Bagging Accuracy: {accuracy_score(y_test, bag.predict(X_test)):.2f}")
    print(f"Boosting Accuracy: {accuracy_score(y_test, boost.predict(X_test)):.2f}")

if __name__ == "__main__":
    run_experiment()
