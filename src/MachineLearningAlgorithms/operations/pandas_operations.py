"""
Python library
Has functions for analyzing, cleaning, exploring and manipulating data
Two-dimensional tabular data structure with labeled axes (rows and columns)
"""
import pandas as pd
import ssl
import numpy as np

# TEMP FIX: Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

#import boston house price data
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(housing)

df = pd.DataFrame(housing.data, columns=housing.feature_names)
print(df.head())

#load data from CSV to data frame
#df = pd.read_csv('file path')

#load data from execl to data frame
#df = pd.read_excel('file path')

#exporting from data frame to CSV
#df.to_csv('boston.csv')
#df.to_excel('boston.csv')

#creating dataframe with random values 20 rows and 10 columns between 0 and 1
random_df = pd.DataFrame(np.random.rand(20,10))
random_df.shape

#Inspecting data frame
boston_df.head() #first 5 rows
boston_df.tail() #last 5 rows
boston_df.info() #gives meta data

boston_df.value_counts('Outcome')
boston_df.groupby('Outcome').mean()

#count of all columns
boston_df.count()

#mean value - columnwise
boston_df.mean()

#find standard deviation - column wise
boston_df.std()

#minimum value
boston_df.min()

#maximum value
boston_df.max()

#all statistical measures for df count/mean/std/min/percentiles/max
boston_df.describe()

#adding a column
boston_df['price'] = boston_df.target
boston_df.head()

#removing a row
boston_df.drop(index=0)

#drop a column
boston_df.drop(columns='ZN', axis=1)

#location row using index value
boston_df.iloc[2]

#locating a particular column
print(boston_df.iloc[:,0]) #first column
print(boston_df.iloc[:,1]) #second column

#postive or negative correlation
boston_df.corr()
