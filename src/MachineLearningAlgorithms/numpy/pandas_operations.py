"""
Python library
Has functions for analyzing, cleaning, exploring and manipulating data
Two-dimensional tabular data structure with labeled axes (rows and columns)
"""
import pandas as pd
import ssl

# TEMP FIX: Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

#import boston house price data
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(housing)

df = pd.DataFrame(housing.data, columns=housing.feature_names)
print(df.head())

#load data from CSV to data frame
#load data from CSV to data frame
#exporting from data frame to CSV
