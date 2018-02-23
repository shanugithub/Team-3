# -- coding: utf-8 --
"""
@author: Adarsh Murthy & Shanu Kumar

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import TransformerMixin

# Creating a class for data cleaning
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

# Method for cleaning dataset
def impute_dataframe(train, test):
    return (DataFrameImputer().fit_transform(train), DataFrameImputer().fit_transform(test))

# Reading train and test dataset
train = pd.read_csv('train.csv')
test  = pd.read_csv('test_2.csv')

# Cleaning dataset
train, test = impute_dataframe(train, test)

'''train = train.fillna(train.median())
test = test.fillna(test.median())'''

#training set features
train_feature1_to_25  = train.iloc[:,  1: 26]
train_return_minus_2_and_1    = train.iloc[:, 26: 28]
train_return_2_to_120 = train.iloc[:, 28:147]
train_return_121_to_180 = train.iloc[:,147:207]
train_return_plus_one   = train.iloc[:,    207]
train_return_plus_two   = train.iloc[:,    208]
train_WeightIntraday   = train.iloc[:,    209]
train_WeightDaily   = train.iloc[:,    210]

#test set features
test_feature    = test.iloc[:,   1: 26]
test_Minus_2_and_1    = test.iloc[:,  26: 28]
test_return_2_to_120  = test.iloc[:,  28:147]

#method to plot graph
def plot_graph(ts):
    plt.plot(np.array(ts))
    plt.ylabel('return')
    plt.show()

#method to plot histogram
def plot_hist(a):
    plt.hist(a, bins=50, normed=True)
    plt.show()

submission = pd.read_csv('sample_submission_2.csv')

#importing KNN 
from sklearn.neighbors import KNeighborsRegressor

# create an KNN model
def knn(X, y):
    model = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=4)
    model.fit(X, y)
    return model

# daily weights
median_WeightDaily  = knn(train_feature1_to_25, train_WeightDaily)
test_WeightDaily  = 1.0 / median_WeightDaily.predict(test_feature)**4 # reverse weight
test_WeightDaily  = test_WeightDaily / test_WeightDaily .mean() # normalize

# Daily prediction
# weighted median method
train_return_plus_one_median = train_return_plus_one.median() * test_WeightDaily
train_return_plus_two_median = train_return_plus_two.median() * test_WeightDaily
submission.loc[60::62,'Predicted'] = train_return_plus_one_median
submission.loc[61::62,'Predicted'] = train_return_plus_two_median

# Intraday prediction
WR = 1.25
test_WeightIntraday  = test_WeightDaily / WR
train_return_121_to_180_median = train_return_121_to_180.median()
for i in range(0, train_return_121_to_180_median.shape[0]):
    submission.loc[i::62,'Predicted'] = train_return_121_to_180_median[i] * test_WeightIntraday * 1000

    # writing predicted results in submission file
    submission.to_csv('sample_submission_2.csv', index=False)

# ploting histogram of predicted values
submission.hist()

# reading the output from csv
data = pd.read_csv('sample_submission_2.csv', usecols=[1])

#plot submission file of only first 200 results 
plt.plot(data.iloc[1:200],color='g',label='Prediction')
plt.legend(loc='best')
plt.show()