#importing the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('D:\\Salary.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#spillitin the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#training the RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=18,random_state=0)
regressor.fit(X_train,y_train)

#predict the new value
y_pred=regressor.predict(X_test)

print(f'{regressor.score(X_test,y_test):0.2%}')