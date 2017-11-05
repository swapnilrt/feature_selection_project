# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):
    model = RandomForestClassifier()
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    m = SelectFromModel(model)
    m.fit(X,y)
    list_name = [];
    for i,rank in enumerate(m.get_support()):
        if(rank==True):
            list_name.append(X.columns[i])

    return list_name
