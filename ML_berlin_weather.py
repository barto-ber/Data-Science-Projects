import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from sklearn.preprocessing import PolynomialFeatures
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

'''LINEAR REGRESSION'''
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

'''POLYNOMIAL FEATURES'''
poly_features = PolynomialFeatures(degree=7, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)

'''LINEAR FEATURES WITH LINEAR REGRESSION (POLYNOMIAL REGRESSION)'''
# linear.fit(X_poly, y_train)

# predictions_poly = linear.predict(X_poly)

# lin_mse_poly = mean_squared_error(y_test, predictions)
# lin_rmse_poly = np.sqrt(lin_mse)
# print("\nMSE of Polynomial Regression model:\n", lin_mse_poly)
# print("\nRMSE of Polynomial Regression model:\n", lin_rmse_poly)

'''RIDGE REGRESSION'''
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

'''RIDGE REGRESSION WITH POLYNOMIAL FEATURES'''
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

'''RANDOM FOREST'''
rnd_clf = RandomForestRegressor(n_jobs=-1)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)

mse_random_forest = mean_squared_error(y_test, y_pred_rf)
rmse_random_forest = np.sqrt(mse_random_forest)
print("\nMSE of Random Forest model:\n", mse_random_forest)
print("\nRMSE of Random Forest model:\n", rmse_random_forest)

'''GRADIENT BOOST REGRESSOR FOR RANDOM FOREST WITH EARLY STOPPING'''
gbrt = GradientBoostingRegressor(learning_rate=1.0)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_test, y_pred)
            for y_pred in gbrt.staged_predict(X_test)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)

y_pred_gbrt = gbrt.predict(X_test)

mse_gbrt = mean_squared_error(y_test, y_pred_gbrt)
rmse_gbrt = np.sqrt(mse_gbrt)
print("\nMSE of Gradient Boosting Reg with Early Stopping model:\n", mse_gbrt)
print("\nRMSE of Gradient Boosting Reg with Early Stopping model:\n", rmse_gbrt)

'''GBR WITH IMPLEMENTATION EARLY STOPPING BY REAL STOPPING EARLY'''
gbrt = GradientBoostingRegressor(warm_start=True)
min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 100):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_test)
    val_error = mean_squared_error(y_test, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break # early stopping
        rmse_gbrt_best = np.sqrt(val_error)
        print("\nMSE of Gradient Boosting Reg with warm start:\n", val_error)
        print("\nRMSE of Gradient Boosting Reg with warm start:\n", rmse_gbrt_best)
        print("\nError going up:\n", error_going_up)


'''EXTREME GRADIENT BOOSTING'''
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
y_pred_xgb = xgb_reg.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
print("\nMSE of Extreme Gradient Boosting Reg model:\n", mse_xgb)
print("\nRMSE of Extreme Gradient Boosting Reg model:\n", rmse_xgb)

xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)
y_pred_xgb = xgb_reg.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
print("\nMSE of Extreme Gradient Boosting Reg model, early stop 10 rounds:\n", mse_xgb)
print("\nRMSE of Extreme Gradient Boosting Reg model, early stop 10 rounds:\n", rmse_xgb)
