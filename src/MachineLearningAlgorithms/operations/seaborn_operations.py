#Data visualization library

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#total bill vs tip dataset
tips = sns.load_dataset('tips')

print(tips.head())
sns.replot(data=tips, x='total_bill', y='tip',  col='time')
