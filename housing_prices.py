import pandas
import xgboost as xgb
import math
from sklearn import preprocessing
import numpy as np
#Load the data

def write_result(df, name, header_list, add_me = True):
    #This functions takes a data frame and writes it to a cs
    df.to_csv(name, header = header_list, index = False)

data = pandas.read_csv('train.csv')
#data['TotalArea'] = data['TotalBsmtSF'] + data['GrLivArea'] + data['GarageArea'] + data['WoodDeckSF'] + data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch'] + data['PoolArea']
test_x = pandas.read_csv('test.csv')
#test_x['TotalArea'] = test_x['TotalBsmtSF'] + test_x['GrLivArea'] + test_x['GarageArea'] + test_x['WoodDeckSF'] + test_x['OpenPorchSF'] + test_x['EnclosedPorch'] + test_x['3SsnPorch'] + test_x['ScreenPorch'] + test_x['PoolArea']
#Get the data and target variable
data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_x.replace([np.inf, -np.inf], np.nan, inplace=True)

train_x = data.iloc[:,:-1]
train_x = train_x.drop(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'LotFrontage', 'GarageYrBlt'], axis = 1)
test_x = test_x.drop(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'LotFrontage', 'GarageYrBlt'], axis = 1)
train_y = data['SalePrice']

######################################
#Start off with a simple xbgoost model
model = xgb.XGBRegressor(n_estimators=100, seed=37, max_depth = 20)
model.fit(train_x, train_y)

from sklearn.metrics import mean_squared_error
y_pred = model.predict(train_x)
print(mean_squared_error(y_pred, train_y))

test_pred = model.predict(test_x)

#Creating pandas data frame
result = pandas.DataFrame({"Id":test_x["Id"], "SalePrice":test_pred})
write_result(result, "Output_xgboost.csv", ["Id", "SalePrice"])

######################################
#Perform a CV grid search to obtain optimal parameters for xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error, make_scorer

# parameters = {"n_estimators" : [n for n in range(5, 100, 5)],
# "max_depth": [depth for depth in range(1, 30)]}
# xgb_model = xgb.XGBRegressor()
# grid = GridSearchCV(xgb_model, parameters,
# n_jobs = 6, cv = 5)
# grid.fit(train_x, train_y)
# grid_result = grid.fit(train_x, train_y)
#
# print(grid_result.best_params_)
# best_params = grid_result.best_params_
# #max depth = 2, n_estimators = 70
#
# model_best_xgb = xgb.XGBRegressor(n_estimators = best_params['n_estimators'],
# max_depth = best_params['max_depth'])
# model_best_xgb.fit(train_x, train_y)
#
# test_pred_best_xgb = model_best_xgb.predict(test_x)
# result = pandas.DataFrame({"Id":test_x["Id"], "SalePrice":test_pred_best_xgb})
# write_result(result, "Output_best_xgboost.csv", ["Id", "SalePrice"])

#This is my baseline performance. This achieves a score of .15256

######################################
#Performing one-hot encoding and then grid searching
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

data = pandas.read_csv('train.csv')
test_x = pandas.read_csv('test.csv')
train_y = np.log(data['SalePrice'])

def hacky_impute(X):
    for name, values in X.iteritems():
        if X[name].dtype == np.dtype(object):
            #an issue with doing this is that most of the non-numeric columns have missing values
            #because that variable is not applicable i.e. there's no pool so there's no PoolQC
            #imp = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
            #imp = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value = "None")
            #temp = imp.fit_transform(X[[name]])
            #X[name] = temp.squeeze()
            X[name] = X[name].fillna("Empty")
        else:
            #print(name)
            imp = SimpleImputer(missing_values = np.nan, strategy='median')
            temp = imp.fit_transform(X[[name]])
            X[name] = temp.squeeze()
    return X

data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_x.replace([np.inf, -np.inf], np.nan, inplace=True)

train_x = data.iloc[:,:-1]
train_x = hacky_impute(train_x)
train_x['TotalIndoor'] = train_x['TotalBsmtSF'] + train_x['GrLivArea']
train_x['TotalOutdoor'] = train_x['GarageArea'] + train_x['WoodDeckSF'] + train_x['OpenPorchSF'] + train_x['EnclosedPorch'] + train_x['3SsnPorch'] + train_x['ScreenPorch'] + train_x['PoolArea']
train_x['TotalBathrooms'] = train_x['BsmtFullBath'] + train_x['BsmtHalfBath'] + train_x['FullBath'] + train_x['HalfBath']

test_x = hacky_impute(test_x)
test_x['TotalIndoor'] = test_x['TotalBsmtSF'] + test_x['GrLivArea']
test_x['TotalOutdoor'] = test_x['GarageArea'] + test_x['WoodDeckSF'] + test_x['OpenPorchSF'] + test_x['EnclosedPorch'] + test_x['3SsnPorch'] + test_x['ScreenPorch'] + test_x['PoolArea']
test_x['TotalBathrooms'] = test_x['BsmtFullBath'] + test_x['BsmtHalfBath'] + test_x['FullBath'] + test_x['HalfBath']

train_categ = train_x.select_dtypes(include = np.dtype(object))
test_categ = test_x.select_dtypes(include = np.dtype(object))

#using get_dummies
features = pandas.concat([train_categ, test_categ], axis = 0)
features = pandas.get_dummies(features)
train_categ = features.iloc[:len(train_x), :]
test_categ = features.iloc[len(train_x):, :]

train_numeric = train_x.select_dtypes(exclude = np.dtype(object))
test_numeric = test_x.select_dtypes(exclude = np.dtype(object))

from scipy import stats
train_outlier = (np.abs(stats.zscore(train_numeric)) < 3).all(axis=1)
train_numeric = train_numeric[train_outlier]
train_categ = train_categ[train_outlier]
train_y = train_y[train_outlier]

train_x = pandas.concat([train_numeric, train_categ], axis = 1, join = 'inner')
test_x = pandas.concat([test_numeric, test_categ], axis = 1, join = 'inner')

train_x['Id'] = train_x['Id'].astype(int)
test_x['Id'] = test_x['Id'].astype(int)
id_test = pandas.DataFrame.copy(test_x)

#Transformations to the features
train_x['LotArea'] = np.log(train_x['LotArea'])
train_x['MasVnrArea'] = np.log1p(train_x['MasVnrArea'])

test_x['LotArea'] = np.log(test_x['LotArea'])
test_x['MasVnrArea'] = np.log1p(test_x['MasVnrArea'])

#Scaling training data
train_names = train_x.columns
test_names = test_x.columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = pandas.DataFrame(scaler.fit_transform(train_x), columns = train_names)
test_x = pandas.DataFrame(scaler.fit_transform(test_x), columns = test_names)

####################
###Get a plot of the features
#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.histplot(train_x[['LotArea']])
#plt.show()

####################
#PCA to see if this helps at all
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 20)
# pca.fit(train_x)
# train_x = (pca.fit_transform(train_x))
# test_x = (pca.transform(test_x))
# print(train_x.shape)
# print(test_x.shape)
####################


# parameters = {"n_estimators" : [n for n in range(5, 100, 5)],
# "max_depth": [depth for depth in range(1, 50)]}
# xgb_model = xgb.XGBRegressor()
# grid = GridSearchCV(xgb_model, parameters,
# n_jobs = 6, cv = 5)
# grid.fit(train_x, train_y)
# grid_result = grid.fit(train_x, train_y)
#
# print(grid_result.best_params_)
# best_params = grid_result.best_params_
# #Values are max_depth = 2, n_estimators = 95 for both log scale and regular y variable
# model_best_xgb = xgb.XGBRegressor(n_estimators = best_params['n_estimators'],
# max_depth = best_params['max_depth'])
model_best_xgb = xgb.XGBRegressor(n_estimators = 95,
max_depth = 2)
model_best_xgb.fit(train_x, train_y)
pred_train = model_best_xgb.predict(train_x)
test_pred_best_xgb = model_best_xgb.predict(test_x)
test_pred_best_xgb = np.exp(test_pred_best_xgb)

print("Training error for xgb ", np.sqrt(mean_squared_error(train_y, pred_train)))
#Calculating the cross validation error for 5 folds
from sklearn.model_selection import cross_val_score
xgb_cross_score = np.mean(np.sqrt(-1*cross_val_score(model_best_xgb, train_x, train_y, cv = 10, scoring = 'neg_mean_squared_error')))
print("CV error for xgb ", xgb_cross_score)
#Get the feature importance from xgboost model
#print(model_best_xgb.feature_importances_)
#from matplotlib import pyplot

#pyplot.bar(range(len(model_best_xgb.feature_importances_)), model_best_xgb.feature_importances_)
#pyplot.show()

#result = pandas.DataFrame({"Id":test_x["Id"], "SalePrice":test_pred_best_xgb})
#write_result(result, "Output_enc_best_xgboost_logscale_StandardScaler.csv", ["Id", "SalePrice"])

######################################
#Implementing random forest
from sklearn.ensemble import RandomForestRegressor
#n_estimators = 100, max_depth = 5 does very well and doesn't have much overfitting
ranfor_model = RandomForestRegressor(n_estimators = 100, max_depth = 5)
ranfor_model.fit(train_x, train_y)
test_pred_ranfor = ranfor_model.predict(test_x)
pred_train_ranfor = ranfor_model.predict(train_x)
print("Random forest error", np.sqrt(mean_squared_error(train_y, pred_train_ranfor)))
ranfor_cross_score = np.mean(np.sqrt(-1*cross_val_score(ranfor_model, train_x, train_y, cv = 10, scoring = 'neg_mean_squared_error')))
print("CV error for RF ", ranfor_cross_score)
######################################
#Implementing an ensemble method with a the best xgboost model and
#a linear regression model
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(train_x, train_y)
test_pred_linear = lin_model.predict(test_x)
pred_train = lin_model.predict(train_x)

#Training a basic neural net
from sklearn.neural_network import MLPRegressor
neural_model = MLPRegressor(hidden_layer_sizes = (100, 10))
neural_model.fit(train_x, train_y)
test_pred_neural = neural_model.predict(test_x)
pred_train_neural = neural_model.predict(train_x)
print(np.sqrt(mean_squared_error(train_y, pred_train_neural)))
neural_cross_score = np.mean(np.sqrt(-1*cross_val_score(neural_model, train_x, train_y, cv = 10, scoring = 'neg_mean_squared_error')))
print("CV error for MLP ", neural_cross_score)

######################################
#Performing ridge regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, ElasticNet

poly = PolynomialFeatures(2)
train_quad_x = poly.fit_transform(train_x)
test_quad_x = poly.fit_transform(test_x)

parameters = {"alpha" : [n/20 for n in range(0, 100)]}
ridge_model = Ridge()
grid = GridSearchCV(ridge_model, parameters,
n_jobs = 6, cv = 5)
grid.fit(train_x, train_y)
grid_result = grid.fit(train_x, train_y)
print(grid_result.best_params_)
##Result is alpha = 0.95 for train_quad_x
##Result is alpha = 4.95 for train_x

best_params = grid_result.best_params_
model_best_ridge = Ridge(alpha = best_params['alpha'])
model_best_ridge.fit(train_x, train_y)
pred_train = model_best_ridge.predict(train_x)
test_pred_ridge = model_best_ridge.predict(test_x)
#transform from log scale back to base 10
test_pred_ridge = np.exp(test_pred_ridge)
#result = pandas.DataFrame({"Id":test_x["Id"], "SalePrice":test_pred_ridge})
#write_result(result, "Output_best_ridge_log.csv", ["Id", "SalePrice"])


#Implementing ElasticNet regression
elastic_model = ElasticNet()
elastic_model.fit(train_x, train_y)
test_pred_elastic = elastic_model.predict(test_x)
pred_train_elastic = elastic_model.predict(train_x)
print("ElasticNet error", np.sqrt(mean_squared_error(train_y, pred_train_elastic)))
elastic_cross_score = np.mean(np.sqrt(-1*cross_val_score(elastic_model, train_x, train_y, cv = 10, scoring = 'neg_mean_squared_error')))
print("CV error for ElasticNet ", elastic_cross_score)

###################################
#svr implementation
from sklearn.svm import SVR
svr_model = SVR()
svr_model.fit(train_x, train_y)
test_pred_svr = svr_model.predict(test_x)
pred_train_svr = svr_model.predict(train_x)
print("SVR error", np.sqrt(mean_squared_error(train_y, pred_train_svr)))
svr_cross_score = np.mean(np.sqrt(-1*cross_val_score(svr_model, train_x, train_y, cv = 10, scoring = 'neg_mean_squared_error')))
print("CV error for SVR ", svr_cross_score)

###################################
#Training a bunch of regressor tree-like algorithms
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor
xtreme_rand_model = ExtraTreesRegressor()
xtreme_rand_model.fit(train_x, train_y)

bagging_model = BaggingRegressor()
bagging_model.fit(train_x, train_y)

ada_model = AdaBoostRegressor()
ada_model.fit(train_x, train_y)

gradient_model = GradientBoostingRegressor()
gradient_model.fit(train_x, train_y)

hist_model = HistGradientBoostingRegressor()
hist_model.fit(train_x, train_y)

####################################
#Implementing LASSO
from sklearn.linear_model import Lasso
lasso_model = Lasso()
lasso_model.fit(train_x, train_y)
test_pred_lasso = lasso_model.predict(test_x)
pred_train_lasso = lasso_model.predict(train_x)
print("LASSO error", np.sqrt(mean_squared_error(train_y, pred_train_lasso)))
lasso_cross_score = np.mean(np.sqrt(-1*cross_val_score(lasso_model, train_x, train_y, cv = 10, scoring = 'neg_mean_squared_error')))
print("CV error for LASSO ", lasso_cross_score)

###################################
#Creating an ensemble
estimators = []
estimators.append(("xgb", model_best_xgb))
estimators.append(("ridge", model_best_ridge))
estimators.append(("random_forest", ranfor_model))
estimators.append(("xtreme", xtreme_rand_model))
estimators.append(("bagging", bagging_model))
estimators.append(("ada", ada_model))
estimators.append(("gradient", gradient_model))
estimators.append(("hist", hist_model))
estimators.append(("svr", svr_model))

from sklearn.ensemble import VotingRegressor
ensem = StackingRegressor(estimators)
test_pred_ensemble = np.exp(ensem.fit(train_x, train_y).predict(test_x))
pred_ensemble = (ensem.fit(train_x, train_y).predict(train_x))
#Make predictions and write to csv
result = pandas.DataFrame({"Id":id_test['Id'], "SalePrice":test_pred_ensemble})
write_result(result, "Output_ensemble_xgb_ridge_ranfor_trees_svr_lasso_newvars_imputing_scaling_outliers_StandardScaler_stacking.csv", ["Id", "SalePrice"])
print("ensemble error", np.sqrt(mean_squared_error(train_y, pred_ensemble)))
ensemble_cross_score = np.mean(np.sqrt(-1*cross_val_score(ensem, train_x, train_y, cv = 10, scoring = 'neg_mean_squared_error')))
print("CV error for ensemble ", ensemble_cross_score)
