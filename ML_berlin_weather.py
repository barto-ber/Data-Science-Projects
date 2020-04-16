import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
pd.options.display.width = 0
pd.options.display.max_rows = None

data = pd.read_csv('berlin_klima_1948_2019_en.txt', sep=';')
less_columns = ["Station_ID", "QN_3", "QN_4", "VPM", "eor", "Average_air_pressure", "Medium_Wind_Speed",
                "Precipitation_form", "Means_of_coverage", "Daily_mean_temp", "Daily_min_temp_ground"]
data.drop(less_columns, inplace=True, axis=1)
data = data.replace(-999.0, np.nan)
# data.fillna(data.median(), inplace=True)
data = data[data['Measurement_date'] >= 19740101]
data.reset_index(inplace=True)
data = data.drop(['index'], axis=1)
# print(data.info())
# print(data[1500:1600])

# Treating infinity as NaN:
pd.set_option('use_inf_as_na', True)
# print("Check -999:\n", (data[(data['Daily_sum_sunshine'] == -999)]).count())
# data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=data.columns, how="all")
data = data.replace([np.inf, -np.inf], 0).dropna(subset=data.columns, how="all")
# data.fillna(data.median(), inplace=True)
# print(data.head())
# print("\nHow many NaN in dataset?\n", data.isnull().sum().sum())
# print("\nNo NaN in dataset:\n", np.all(np.isfinite(data)))
# attributes = ['Measurement_date', 'Max_Wind_Speed', 'Precipitation_level', 'Daily_sum_sunshine', 'Daily_snow_depth',
#               'Daily_mean_humidity', 'Daily_max_temp', 'Daily_min_temp']
# scatter_matrix(data[attributes])
# plt.show()

predict = "Daily_max_temp"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print("\nper-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
# print("\nper-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))

linear = linear_model.LinearRegression()

linear.fit(X_train_scaled, y_train)

acc_train = linear.score(X_train_scaled, y_train)
acc_test = linear.score(X_test_scaled, y_test)
print("\nScore for linear regression training set:\n", acc_train)
print("\nScore for linear regression test set:\n", acc_test)

predictions = linear.predict(X_test_scaled) # Gets a list of all predictions

lin_mse = mean_squared_error(y_test, predictions)
lin_rmse = np.sqrt(lin_mse)
print("\nMSE of Linear Regression model:\n", lin_mse)
print("\nRMSE of Linear Regression model:\n", lin_rmse)


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=10, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)
# linear.fit(X_poly, y_train)

# predictions_poly = linear.predict(X_poly)

# lin_mse_poly = mean_squared_error(y_test, predictions)
# lin_rmse_poly = np.sqrt(lin_mse)
# print("\nMSE of Polynomial Regression model:\n", lin_mse_poly)
# print("\nRMSE of Polynomial Regression model:\n", lin_rmse_poly)


linear_ridge = linear_model.Ridge()
linear_ridge.fit(X_train_scaled, y_train)
acc_ridge_train = linear_ridge.score(X_train_scaled, y_train)
acc_ridge_test = linear_ridge.score(X_test_scaled, y_test)
print("\nScore for ridge regression training set:\n", acc_ridge_train)
print("\nScore for ridge regression test set:\n", acc_ridge_test)

predictions_ridge = linear_ridge.predict(X_test_scaled)

lin_mse_ridge = mean_squared_error(y_test, predictions_ridge)
lin_rmse_ridge = np.sqrt(lin_mse_ridge)
print("\nMSE of Ridge Regression model:\n", lin_mse_ridge)
print("\nRMSE of Ridge Regression model:\n", lin_rmse_ridge)


scaler.fit(X_train_poly)
X_train_poly_scaled = scaler.transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

linear_ridge_poly = linear_model.Ridge()
linear_ridge_poly.fit(X_train_poly_scaled, y_train)
# acc_ridge_train_poly = linear_ridge.score(X_poly_scaled, y_train)
# acc_ridge_test_poly = linear_ridge.score(X_test_scaled, y_test)
# print("\nScore for ridge regression training set:\n", acc_ridge_train)
# print("\nScore for ridge regression test set:\n", acc_ridge_test)

predictions_ridge_poly = linear_ridge_poly.predict(X_test_poly_scaled)

lin_mse_ridge_poly = mean_squared_error(y_test, predictions_ridge_poly)
lin_rmse_ridge_poly = np.sqrt(lin_mse_ridge_poly)
print("\nMSE of Polynomial Ridge Regression model:\n", lin_mse_ridge_poly)
print("\nRMSE of Polynomial Ridge Regression model:\n", lin_rmse_ridge_poly)

# for x in range(50):
#     print(predictions[x], X_test[x], y_test[x])






# plot = "Measurement_date"
# plt.scatter(data[plot], data["Daily_max_temp"])
# plt.legend(loc=4)
# plt.xlabel(plot)
# plt.ylabel("Daily max temp")
# plt.show()