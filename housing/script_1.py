# import libraries
import pandas as pd
import numpy as np

# read data
df_train=pd.read_csv('./train.csv')
df_test=pd.read_csv('./test.csv')

#store Ids of homes
df_train=df_train.drop('Id', axis=1)
y_id=df_test['Id'].copy()
df_test=df_test.drop('Id', axis=1)

#define y_train
y_train=df_train['SalePrice'].values.reshape(-1,1)
df_train=df_train.drop('SalePrice', axis=1)

#concate df_train and df_test
df=pd.concat([df_train, df_test], axis=0, ignore_index=True)

#select columns with non null values
df=df.dropna(axis=1)

#transform categorical variables into dummy variables
df=pd.get_dummies(df, drop_first=True)

#create X_train and X_test
X_train=df.iloc[:df_train.shape[0],]
X_test=df.iloc[df_train.shape[0]:,]

# Pipeline and Regression

# import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# steps
steps = [('scaler', StandardScaler()),
         ('ridge', Ridge())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'ridge__alpha':np.logspace(-4, 0, 50)}

# Create the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict y_pred
y_pred=cv.predict(X_test)

#shape to export
output=pd.concat([y_id, DataFrame(y_pred)], axis=1, ignore_index=True)
output.columns=['Id', 'SalePrice']

#export
output.to_csv('./submission_1.csv', sep=',', index=False)
