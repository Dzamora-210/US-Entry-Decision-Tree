import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import warnings

#import data 
df = pd.read_csv('/Users/dantezamora/Downloads/Border_Crossing_Entry_Data.csv')
df.head(5)
df.info

#check for null values in training data
print('Shape: ')
print(df.shape)
print('Distribution: ')
df.isnull().sum(axis=0)

#extract necessary columns
df_train = df.copy()
col_to_drop = ['Port Code', 'Location']
df_train = df_train.drop(columns = col_to_drop)

#create weekend and time of day columns
#determine min and max dates
df_train['Date'] = pd.to_datetime(df_train['Date'])
earliest_date = df['Date'].min()
latest_date = df['Date'].max()
print(f'Earliest Date: {earliest_date}')
print(f'Latest Date: {latest_date}')

#weekend column
df_train['Weekend'] = df_train['Date'].dt.dayofweek.isin([5,6])

df_train.head()

#value check
df_train['Weekend'].value_counts()

#view most popular entry points
top10 = (df_train['Port Name'].value_counts()).iloc[:10]

top10.plot(kind = 'bar')

#create train-test split
X = df_train[['Port Name', 'Border', 'Date', 'Measure', 'Value', 'Weekend']]
y = df_train[['State']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#turning categorical training data to numerical with OneHotEncoder
oh = OneHotEncoder()

encoded_data = oh.fit_transform(X_train[['Port Name','Border','Measure','Value', 'Weekend']])

encoded_df = pd.DataFrame(encoded_data.toarray(), columns=oh.get_feature_names_out(['Port Name','Border','Measure','Value', 'Weekend']))

encoded_df.isnull().sum(axis=0)

#create list of column names to organize encoded array 
onehot_col_names = list(oh.get_feature_names_out(col_to_encode.columns))
print(onehot_col_names)

X_train = oh.transform(df_train)

#check dimensions
print(df_train.shape)
print(X_train.shape)
print(y_train.shape)

#create decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

tree_onehot_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=4)
tree_onehot_entropy.fit(X_train, y_train)

fig = plt.figure(figsize(15,10))
ax = fig.gca()
plot_tree(tree_onehot_gini, feature_names=onehot_col_names, filled=True, ax=ax)