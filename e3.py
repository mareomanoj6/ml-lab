import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("diabetes.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear = LinearRegression()
linear.fit(X_train, y_train)

y_pred = linear.predict(X_test)

print("\nLinear Regression")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)

print("\nRidge")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)

print("\nLasso")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
