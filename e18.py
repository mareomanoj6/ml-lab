import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

def run_experiment(df, features, title):
    X = df[features]
    y = df['medv']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_err, val_err = [], []
    for d in range(1, 9):
        model = make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression()).fit(X_train, y_train)
        t_err = mean_squared_error(y_train, model.predict(X_train))
        v_err = mean_squared_error(y_val, model.predict(X_val))
        train_err.append(t_err)
        val_err.append(v_err)

    print(f"\n--- {title} ---")
    print(f"{'Degree':<10}{'Train Error':<20}{'Val Error':<20}")
    for d, te, ve in zip(range(1, 9), train_err, val_err):
        print(f"{d:<10}{te:<20.4f}{ve:<20.4f}")
    
    return train_err, val_err

if __name__ == "__main__":
    df = pd.read_csv('boston_housing.csv')
    
    t1, v1 = run_experiment(df, ['rm'], "1 Feature ('rm')")
    t2, v2 = run_experiment(df, [c for c in df.columns if c != 'medv'], "All Features")
    
    plt.figure()
    plt.plot(range(1, 9), t1, label='1-Feat Train')
    plt.plot(range(1, 9), v1, label='1-Feat Val')
    plt.plot(range(1, 9), t2, label='All-Feat Train')
    plt.plot(range(1, 9), v2, label='All-Feat Val')
    plt.xlabel('Degree')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.title('Bias-Variance Tradeoff: 1 vs All Features')
    plt.show()
