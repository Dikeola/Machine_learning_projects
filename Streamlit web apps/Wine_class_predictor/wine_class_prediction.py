import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pickle

wine = load_wine()
wine_x = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_y = pd.Series(wine.target)

X_train, X_test, y_train, y_test = train_test_split(wine_x, wine_y, test_size=.2, random_state=1)

rfc = RandomForestClassifier()
rfc.fit(X_train[['proline', 'flavanoids']], y_train)

pickle.dump(rfc, open('wine_class_predictor.pkl', 'wb'))
