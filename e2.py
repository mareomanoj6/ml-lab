import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("auto_mpg.csv")

df = df[["mpg", "displacement"]].dropna()

X = df[["displacement"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

linear = LinearRegression()
linear.fit(X_train, y_train)

y_pred = linear.predict(X_test)

print("\nLinear Regression")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

for degree in [2, 3, 4, 5]:
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nPolynomial Degree:", degree)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

x_plot = np.linspace(
    X["displacement"].min(),
    X["displacement"].max(),
    200
).reshape(-1, 1)

y_plot = model.predict(x_plot)

plt.scatter(X, y, color="blue")
plt.plot(x_plot, y_plot, color="red")

plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Polynomial Regression")
plt.show()

