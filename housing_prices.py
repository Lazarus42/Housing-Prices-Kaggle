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
test_x = pandas.read_csv('test.csv')
#Get the data and target variable
data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_x.replace([np.inf, -np.inf], np.nan, inplace=True)

train_x = data.iloc[:,:-1]
train_x = train_x.drop(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'LotFrontage', 'GarageYrBlt'], axis = 1)
test_x = test_x.drop(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'LotFrontage', 'GarageYrBlt'], axis = 1)
train_y = data['SalePrice']

#Start off with a simple xbgoost model
model = xgb.XGBRegressor(n_estimators=101, seed=37, max_depth = 20)
model.fit(train_x, train_y)

from sklearn.metrics import mean_squared_error
y_pred = model.predict(train_x)
print(mean_squared_error(y_pred, train_y))




test_pred = model.predict(test_x)

#Creating pandas data frame
result = pandas.DataFrame({"Id":test_x["Id"], "SalePrice":test_pred})
write_result(result, "Output.csv", ["Id", "SalePrice"])
