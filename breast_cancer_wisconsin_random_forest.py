import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('./data.csv')
print(data.shape)

data.head()
data = data.drop(['id', 'Unnamed: 32'], axis = 1)
data.describe()

plt.rcParams['figure.figsize'] = (22, 12)

corr_matrix = data.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(corr_matrix)] = False

sns.heatmap(corr_matrix, mask=mask, cmap = 'pink', annot = True, linewidths = 0.5, fmt = '.1f', square = False)
plt.title('Mapa de calor para correlações', fontsize = 20)
plt.show()

y = data['diagnosis']

x = data.drop('diagnosis', axis = 1)
x = (x - x.mean()) / (x.std())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, train_size = 0.7, random_state = 16)

print("Shape of x_train :", x_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_test :", y_test.shape)

plt.style.use('fivethirtyeight')

model = RandomForestClassifier(n_estimators = 400, max_depth = 10)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Training accuracy :", model.score(x_train, y_train))
print("Testing accuarcy :", model.score(x_test, y_test))

cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, cmap = 'winter')
plt.title('Confusion Matrix', fontsize = 20)
plt.show()

model = RandomForestClassifier() 
rfecv = RFECV(estimator = model, step = 1, cv = 5, scoring = 'accuracy')
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])

y_pred = rfecv.predict(x_test)

print("Training Accuracy :", rfecv.score(x_train, y_train))
print("Testing Accuracy :", rfecv.score(x_test, y_test))

cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, cmap = 'winter')
plt.title('Confusion Matrix', fontsize = 20)
plt.show()

list_to_delete = ['perimeter_mean','radius_mean',
    'compactness_mean','concave points_mean','radius_se',
    'perimeter_se','radius_worst','perimeter_worst',
    'compactness_worst','concave points_worst',
    'compactness_se','concave points_se','texture_worst','area_worst']

x = x.drop(list_to_delete, axis = 1)

plt.rcParams['figure.figsize'] = (18, 15)

corr_matrix = x.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(corr_matrix)] = False

sns.heatmap(corr_matrix, mask=mask, cmap = 'pink', annot = True, linewidths = 0.5, fmt = '.1f')
plt.title('Mapa de calor para correlação com aperfeiçoamento', fontsize = 20)
plt.show()

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, train_size = 0.7, random_state = 16)

print("Shape of x_train :", x_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_test :", y_test.shape)

model = RandomForestClassifier(n_estimators = 400, max_depth = 10)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Training accuracy :", model.score(x_train, y_train))
print("Testing accuarcy :", model.score(x_test, y_test))


cr = classification_report(y_test, y_pred)
print(cr)

cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, cmap = 'winter')
plt.title('Confusion Matrix', fontsize = 20)
plt.show()