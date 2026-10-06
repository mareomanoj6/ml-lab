import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

def run_experiment():
    df = pd.read_csv('boston_housing.csv')
    X = df[['rm']] # Use 'rm' (rooms) as feature
    y = df['medv']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_err, val_err = [], []
    for d in range(1, 6):
        model = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(X_train, y_train)
        train_err.append(mean_squared_error(y_train, model.predict(X_train)))
        val_err.append(mean_squared_error(y_val, model.predict(X_val)))

    plt.plot(range(1, 6), train_err, label='Train')
    plt.plot(range(1, 6), val_err, label='Val')
    plt.xlabel('Degree')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Bias-Variance Tradeoff')
    plt.savefig('e18.png')
    print("Plot saved as e18.png")

if __name__ == "__main__":
    run_experiment()
