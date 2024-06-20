# Bitcoin-price-prediction-project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from datetime import datetime
 import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r'C:\Users\pilli.pavithra\Documents\bitcoin\bitcoin_2017_to_2023.csv')
data.head()
data['timestamp']
data.isnull().sum()
data.describe()

data['year'] = pd.to_datetime(data['timestamp']).dt.year

# extracting year data and displaying
print(data[['timestamp', 'year']])
sns.relplot(data=data, x='year', y='close', kind='line')
