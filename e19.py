import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_experiment():
    df = pd.read_csv('adult_income.csv')
    # Basic preprocessing
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    X = df.drop('income', axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression().fit(X_train_s, y_train)
    dt = DecisionTreeClassifier().fit(X_train, y_train)

    print(f"LogReg Acc: {accuracy_score(y_test, lr.predict(X_test_s)):.2f}")
    print(f"DecTree Acc: {accuracy_score(y_test, dt.predict(X_test)):.2f}")

if __name__ == "__main__":
    run_experiment()
