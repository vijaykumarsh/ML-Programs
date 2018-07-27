
# coding: utf-8

# In[120]:

##The purpose of the project is to predict median house values in Californian districts, given many features from these districts

##import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##import the data
data=pd.read_csv('C:\\Users\\vsn8kor\\Downloads\\Machine_Learning_Projects (1)\\Projects\\Projects for submission\\California Housing Price Prediction\\Dataset for the project\\housing.csv')
#pd.DataFrame(data)


# In[121]:

##Getting the IV,DV

iv=data.iloc[:,0:9].values
#print(pd.DataFrame(iv))
#print(pd.DataFrame(iv))
dv=data[['median_house_value']].values
#print(pd.DataFrame(dv))
#pd.DataFrame(dv)


# In[122]:

##Filling NA with mean of the values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(iv[:,0:8])
iv[:,0:7]=imputer.fit_transform(iv[:,0:7])
#print(pd.DataFrame(iv[:,0:8]))


# In[123]:

##Performing One Hot Encoder 
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# lbl=LabelEncoder()
# iv['ocean_proximity']=lbl.fit_transform(iv['ocean_proximity'])
# onehotencoder = OneHotEncoder(categorical_features=[9])
# iv = onehotencoder.fit_transform(iv).toarray()
##Get dummies
iv_dummies=pd.get_dummies(iv[:,8])
iv=pd.DataFrame(iv[:,0:7])

iv=pd.concat([iv,iv_dummies],axis=1)
#print(iv)


# In[124]:

##Splitting the data set into Test and Train (80-20)
from sklearn.cross_validation import train_test_split
iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=0)


# In[125]:

##Import standard scaler
##Performing Standard Scaler to - remove dominance

from sklearn.preprocessing import StandardScaler
Scale=StandardScaler()

iv_train=Scale.fit_transform(iv_train) ##Fit and Transform
iv_test=Scale.transform(iv_test) ##Only Transform


# In[126]:

##Performing Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
regressor=LinearRegression()
##Fit train
regressor.fit(iv_train,dv_train)
y_pred=regressor.predict(iv_test)
print('Accuracy of LR',mean_squared_error(y_pred,dv_test))


# In[127]:

#performing Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
DTRegressor=DecisionTreeRegressor(random_state=0)
DTRegressor.fit(iv_train,dv_train)

##Predicting output
y_pred_DT=DTRegressor.predict(iv_test)
print('Accuracy of DT',mean_squared_error(y_pred_DT,dv_test))


# In[128]:

##Performing Random Forest Regression

from sklearn.ensemble import RandomForestRegressor
RFRegressor=RandomForestRegressor()
RFRegressor.fit(iv_train,dv_train)
y_pred_RF=RFRegressor.predict(iv_test)
print('Accuracy of RFR',mean_squared_error(y_pred_RF,dv_test))


# In[129]:

#==============================================================================
# Create confusion matrix to evaluate performance of data
#==============================================================================
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix (dv_test, y_pred)

print(confusionMatrix)

