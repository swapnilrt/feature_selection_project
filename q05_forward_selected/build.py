# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()

def forward_selected (df,model):
    Variable_1= []
    Variable_2 =[]
    np.random.seed(6)
    features = df.iloc[:,:-1]
    target = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.3)
    features = ['OverallQual', 'GrLivArea','BsmtFinSF1', 'GarageCars', 'KitchenAbvGr', '1stFlrSF','YearRemodAdd','LotArea', 'MasVnrArea', 'WoodDeckSF']
    for i in features:
        Variable_1.append (i)
        model.fit(X_train[Variable_1],y_train)
        y_pred = model.predict(X_test[Variable_1])
        acc = r2_score(y_test,y_pred)
        if not Variable_2:
            Variable_2.append (acc)
        elif acc > Variable_2[-1]:
            Variable_2.append(acc)
        else:
            Variable_1.remove(i)
    return Variable_1,Variable_2
