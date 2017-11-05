# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    model = RandomForestClassifier()
    X, y = data.iloc[:,:-1], data.iloc[:,-1]
    nf= X.shape[1]/2
    rfe = RFE(model, n_features_to_select=nf)
    rfe = rfe.fit(X, y)
    lst = [];
    for i,rank in enumerate(rfe.ranking_):
        if(rank==1):
            lst.append(X.columns[i])

    return lst
