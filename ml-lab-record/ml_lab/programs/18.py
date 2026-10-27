from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd

boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

degrees = [1, 2, 3, 4, 5]
tr_err, val_err = [], []
rmse_tr_err, rmse_val_err = [], []
for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_train_p = poly.fit_transform(X_train)
    X_val_p = poly.transform(X_val)

    scaler = StandardScaler()
    X_train_p = scaler.fit_transform(X_train_p)
    X_val_p = scaler.transform(X_val_p)

    model = LinearRegression()
    model.fit(X_train_p, y_train)
    train_pred = model.predict(X_train_p)
    val_pred = model.predict(X_val_p)

    tr_err.append(mean_squared_error(y_train, train_pred))
    val_err.append(mean_squared_error(y_val, val_pred))
    rmse_tr_err.append(root_mean_squared_error(y_train, train_pred))
    rmse_val_err.append(root_mean_squared_error(y_val, val_pred))
results = pd.DataFrame({
    "Degree": degrees,
    "Training Error": tr_err,
    "Validation Error": val_err,
    "Training Error (RMSE)": rmse_tr_err,
    "Validation Error (RMSE)": rmse_val_err
})
numeric = ["Training Error", "Validation Error", "Training Error (RMSE)", "Validation Error (RMSE)"]
results[numeric] = results[numeric].map(lambda x: f"{x:.4f}")
print(results.to_string(index=False))


plt.figure(figsize=(7, 5))
plt.plot(degrees, tr_err, marker='o', label="Training Error (MSE)", color='deeppink')
plt.plot(degrees, val_err, marker='s', label="Validation Error (MSE)", color='slateblue')

plt.xlabel("Degree")
plt.ylabel("Error")
plt.title("Bias-Variance Tradeoff")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()
