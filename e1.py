import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("california_housing.csv")

X = df[["MedInc"]].values
y = df["MedHouseVal"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_ne = np.c_[np.ones(len(X_train)), X_train]

theta = np.linalg.inv(
    X_train_ne.T @ X_train_ne
) @ X_train_ne.T @ y_train

X_test_ne = np.c_[np.ones(len(X_test)), X_test]

y_pred = X_test_ne @ theta

print("Normal Equation")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

X_gd = np.c_[np.ones(len(X_train)), X_train]

theta = np.zeros(2)

learning_rate = 0.01
iterations = 1000

for i in range(iterations):
    prediction = X_gd @ theta
    gradient = (2 / len(X_gd)) * X_gd.T @ (prediction - y_train)
    theta = theta - learning_rate * gradient

X_test_gd = np.c_[np.ones(len(X_test)), X_test]

y_pred = X_test_gd @ theta

print("\nGradient Descent")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))


x_plot = np.linspace(
    X.min(),
    X.max(),
    100
).reshape(-1, 1)
y_plot = np.c_[np.ones(len(x_plot)), x_plot] @ theta

plt.scatter(X_test, y_test, color="blue")
plt.plot(x_plot, y_plot, color="red")

plt.xlabel("MedInc")
plt.ylabel("MedHouseVal")
plt.title("Linear Regression")
plt.show()
