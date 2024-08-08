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
data = pd.read_csv(r"C:\Users\pavit\Downloads\bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv\bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")
data.columns = data.columns.str.lower()
data.head()
data.isnull().sum()
data.describe()
# Convert Unix timestamp to formatted date string
data['date'] = pd.to_datetime(data['timestamp'], unit='s').dt.strftime('%d-%m-%Y')

# Convert formatted date string to datetime object
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')

# Extract year from datetime object
data['year'] = data['date'].dt.year
data = data.drop(columns = ['volume_(btc)', 'volume_(currency)', 'weighted_price', 'timestamp'])
data = data.dropna()
data.columns
fig, ax = plt.subplots(figsize=(10, 5)) 
data.plot.line(y='close',x='date', ax = ax )
fig, ax = plt.subplots(figsize=(10, 5)) 
data.plot.line(y='open',x='date', ax = ax )
fig, ax = plt.subplots(figsize=(10, 5)) 
data.plot.line(y='high',x='date', ax = ax )
fig, ax = plt.subplots(figsize=(10, 5)) 
data.plot.line(y='low',x='date', ax = ax )
data['open-close']  = data['open'] - data['close']
data['low-high']  = data['low'] - data['high']
data['target'] = np.where(data['close'].shift(-1) >data['close'], 1, 0)
#checcking if the target is balanced or not
plt.pie(data['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.show()
#checking for correlated features
plt.figure(figsize=(10, 10))
sns.heatmap(data.corr() > 0.9, annot=True, cbar=False)
plt.show()
averages = data.groupby('year').agg({
    'open': 'mean',
    'high': 'mean',
    'low': 'mean',
    'close': 'mean'
})
print(averages)
plt.subplots(figsize=(20,10))
for i, col in enumerate(['open', 'high', 'low', 'close']):
    plt.subplot(2,2,i+1)
    averages[col].plot.bar()
    plt.title(col.capitalize())
    plt.legend([col.capitalize()])

plt.tight_layout()    
plt.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data.head()
factors = data[['open-close', 'low-high']]
target = data['target']
scaler = StandardScaler()
data_new = scaler.fit_transform(factors)
X_train, X_test, Y_train, Y_test = train_test_split(
    factors, target, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True, verbose=True),  # Verbose output
    XGBClassifier(verbose=1)  # Verbose output
]

for i in range(len(models)):
    print(f'Training model: {models[i]}')
    models[i].fit(X_train, Y_train)
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_test, models[i].predict_proba(X_test)[:,1]))
    print()
