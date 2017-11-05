# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

def percentile_k_features(df,k=20):
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    fs = SelectPercentile(f_regression, percentile=20)
    X_train_fs = fs.fit_transform(X, y)
    return [df.columns[i] for i in np.argsort(fs.scores_)[::-1]][0:7]
