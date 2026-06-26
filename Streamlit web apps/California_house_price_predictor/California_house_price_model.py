import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import xgboost as xgb

cali = fetch_california_housing()
cali_x = pd.DataFrame(cali.data, columns=cali.feature_names)
cali_y= pd.Series(cali.target)

cali_x = cali_x[['MedInc']]

X_train, X_test, y_train, y_test = train_test_split(cali_x, cali_y, test_size=.2, random_state=1)

xgbr = xgb.XGBRegressor()
xgbr.fit(X_train, y_train)

import pickle

pickle.dump(xgbr, open('california_house_price_model.pkl','wb'))