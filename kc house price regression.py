# Assignment 1 - KC House Price Prediction

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import xgboost as xgb


# Set the current working directory
os.chdir('C:\\Users\\minhd\\Google Drive\\CLASS\\KA AI Bootcamp\\Assignments\\Assignment 1')
cwd = os.getcwd()
cwd

# Importing the dataset
dataset = pd.read_csv('kc_house_data.csv')

# Remove ID features
dataset = dataset.drop(["id", "date", "zipcode", "lat", "long"], axis = 1)

# Create X and y variables
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Pre-Processing
#Check if there is any missing value
dataset.isnull().values.any()
#No missing value

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


###--- Multiple Linear Regression ---###

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
results = cross_val_score(estimator = lin_reg, X = X_train, y = y_train, cv = 10)
print("R Squared:", results.mean(), results.std())
# R Squared: 0.6505287134603113 0.01907986081873545 

# Predicting the Test set results
y_pred = lin_reg.predict(X_test)

# Get performance results of Multiple Linear Regression
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) # RMSE = 205420.10018153643
print(np.sqrt(metrics.r2_score(y_test, y_pred))) # R-Squared = 0.8035251775726825


###--- Decision Tree Regression ---###

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressorDT = DecisionTreeRegressor(random_state = 0)
regressorDT.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
resultsDT = cross_val_score(estimator = regressorDT, X = X_train, y = y_train, cv = 10)
print("R Squared:", resultsDT.mean(), resultsDT.std())
# R Squared: 0.49238925503989117 0.06765095338922864

# Predicting the Test set results
y_pred = regressorDT.predict(X_test)

# Get performance results of Decision Tree Regression
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) # RMSE = 237169.32306901756
print(np.sqrt(metrics.r2_score(y_test, y_pred))) # R-Squared = 0.7263979421924547


###--- Random Forest Regression ---###

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressorRF = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressorRF.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
resultsRF = cross_val_score(estimator = regressorRF, X = X_train, y = y_train, cv = 10)
print("R Squared:", resultsRF.mean(), resultsRF.std())
# R Squared: 0.7244648713080072 0.03466646736804159

# Predicting the Test set results
y_pred = regressorRF.predict(X_test)

# Get performance results of Random Forest Regression
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) # RMSE = 178257.19208897787
print(np.sqrt(metrics.r2_score(y_test, y_pred))) # R-Squared = 0.8562524958271753


###--- Support Vector Regression ---###

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressorSVR = SVR(kernel = 'rbf')
regressorSVR.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
resultsSVR = cross_val_score(estimator = regressorSVR, X = X_train, y = y_train, cv = 10)
print("R Squared:", resultsSVR.mean(), resultsSVR.std())
# R Squared: -0.056799047466620366 0.012690372522549367

# Predicting the Test set results
y_pred = regressorSVR.predict(X_test)

# Get performance results of Support Vector Regression
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) # RMSE = 352913.7411019169
print(np.sqrt(metrics.r2_score(y_test, y_pred))) # R-Squared = NaN


###--- Polynomial Regression ---###
      
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# Fitting XGBoost to the Training set
from xgboost import XGBRegressor
regressorXBG = XGBRegressor()
regressorXBG.fit(X_poly, y)

# Predicting the Test set results
y_pred = regressorXBG.predict(X_poly)

# Get performance results of Polynomial Regression
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y, y_pred))) # RMSE = 158623.42410175377
print(np.sqrt(metrics.r2_score(y, y_pred))) # R-Squared = 0.9019692715140315