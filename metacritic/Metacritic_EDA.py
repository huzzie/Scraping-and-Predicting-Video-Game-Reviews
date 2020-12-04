#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
import re

#%% [markdown]
## Import dataset

#%%
path = r"/Users/HyunjaeCho/Documents/GitHub/Team-Amazing/metacritic_review"
all_files = glob.glob(path + '/*.csv')

# 
df = [pd.read_csv(f, index_col =0) for f in all_files]
# extract genres
f_names = [i.split('metacritic_review_',1)[1] for i in all_files]
path = r"/Users/HyunjaeCho/Documents/GitHub/Team-Amazing/metacritic_review"
all_files = glob.glob(path + '/*.csv')
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col= 0, header = 0)
    li.append(df)

total_df = pd.concat(li, axis = 0, ignore_index = True)


#%%
# make genre dictionary
genre = list(total_df['genre'].unique())
values = list(range(0,len(genre)))
genre_dict = dict(zip(genre, values))
print(genre_dict)
# %%
# make platform dictionary
total_df['platform'] = [x.strip(' ') for x in total_df['platform']]
platforms = list(total_df['platform'].unique())
platforms = [x.strip(' ') for x in platforms]
values = list(range(0, len(platforms)))
platform_dict = dict(zip(platforms, values))
print(platform_dict)
# %%
# convert to numerical values
total_df['nplatform'] = total_df['platform'].map(platform_dict)
total_df['ngenre'] = total_df['genre'].map(genre_dict)


#%%
## describe the number of genre and platform
total_df['genre'].value_counts()
total_df['platform'].value_counts()

#%% 
# 3DS, PC, Playstation3, Playstation4, Playstation Vita, 
# Nintendo Switch, Wii, Wii U, Xbox 360, Xbox One
# subset

df = total_df[total_df['platform'].isin(['3DS', 'PC', 'PlayStation 3', 'PlayStation 4',
'PlayStation Vita', 'Switch', 'Wii', 'Wii U', 'Xbox 360', 'Xbox One'])]
total_df = df


#%%
import seaborn as sns
sns.set_theme()
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.countplot(y='genre', data=df, order = df['genre'].value_counts().index)

#%%
sns.boxplot(data=df, x='score', y='genre', order = df['genre'].value_counts().index)
#%%
sns.countplot(y='platform', data=df, order = df['platform'].value_counts().index)
#%%
sns.boxplot(data=df, x='score', y='platform', order = df['platform'].value_counts().index)


#%% [markdown]
## Average plots
# %% 
## genre plot
genre_avg = total_df.groupby('genre').mean()
genre_avg['genre'] = genre_avg.index
graph = sns.barplot(y = 'genre', x = 'score', data = genre_avg)
plt.show()
#%%
# platform plot
platform_avg = total_df.groupby('platform').mean()
platform_avg['platform'] = platform_avg.index
graph = sns.barplot(y = 'platform', x = 'score', data = platform_avg)

#%% [markdown]
## Linear models

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
#%%
#simple linear model
fit = ols(formula = 'score ~ C(nplatform) + C(ngenre)', data = total_df).fit()
print(fit.summary())
# rsq is 0.027
# rsqure is too low. Let's split that test/train data and check what happens
# %%
# set dependent and independent variables
x = total_df[['nplatform', 'ngenre']]
y = total_df['score']

# %% [markdown]
## Linear Model

#%%
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state=1)
full_split1 = linear_model.LinearRegression()
fit = full_split1.fit(xtrain, ytrain)
y_pred1 = fit.predict(xtest)

print('score (train):', '%.4f' % full_split1.score(xtrain, ytrain))
print('score (test):','%.4f' % full_split1.score(xtest, ytest))
print('intercept:','%.4f' % full_split1.intercept_)
print('coef_:', full_split1.coef_)
print('cross evaluation:', cross_val_score(fit, x, y, cv = 10))

# score (train): 0.0077
# score (test): 0.0066
# intercept: 67.2912
# coef_: [ 0.20208439 -0.02632149]
# cross evaluation: [-8.00750758e-01 -1.10802425e-01 -2.35084842e+01 -5.87163438e+00
# -2.51259281e+00 -1.10628290e+00  1.52461781e-02 -2.32306277e-02
# -7.35851228e-03  1.02781806e-02]

# %% [markdown]
## Logistic Model

#%%
logit = LogisticRegression()
fit2 = logit.fit(xtrain, ytrain)
print('score (train):', '%.4f' % logit.score(xtrain, ytrain))
print('score (train):', '%.4f' % logit.score(xtest, ytest))

# score (train): 0.0365
# score (train): 0.0330
#%%
#print(logit.predict_proba(xtrain[:1]))
# too low 
# %%
# make new cut off points 
conditions = [(total_df['score'] < 80), total_df['score'] > 79.9]
values = [0, 1]
total_df['score8'] = np.select(conditions, values)
print(total_df.head())
print(total_df.tail())
# %%
# set new dependent and independent variables
x = total_df[['nplatform', 'ngenre']]
y = total_df['score8']
# split new
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 1)
scorelogit = LogisticRegression()
fit3 = scorelogit.fit(xtrain, ytrain)
y_predit = fit3.predict(xtest)

print('score (train):', '%.4f' % scorelogit.score(xtrain, ytrain))
print('score (test):','%.4f' % scorelogit.score(xtest, ytest))
print('intercept:','%.4f' % scorelogit.intercept_)
print('coef_:', scorelogit.coef_)
print('cross evaluation:', cross_val_score(fit3, x, y, cv = 10))
print("classification report:", classification_report(ytest, y_predit))
print('confusion matrix:', confusion_matrix(ytest, y_predit ))

# score (train): 0.7716
# score (test): 0.7642
# intercept: -1.3878
# coef_: [[0.00436568 0.02577564]]
# cross evaluation: [0.76972909 0.76972909 0.76972909 0.76972909 0.76972909 0.76972909
#  0.76972909 0.76972909 0.76972909 0.77018268]
# classification report:               precision    recall  f1-score   support

#            0       0.76      1.00      0.87      3244
#            1       0.00      0.00      0.00      1001

#     accuracy                           0.76      4245
#    macro avg       0.38      0.50      0.43      4245
# weighted avg       0.58      0.76      0.66      4245

# confusion matrix: [[3244    0]
#  [1001    0]]


# %% [markdown]
## Roc-Curve







#%% [markdown]
## KNN
#%%
from sklearn.preprocessing import scale
xsa = pd.DataFrame(scale(x), columns = x.columns)
ysa = y.copy()

knn_scv = KNeighborsClassifier(n_neighbors= 7)
fit4 = knn_scv.fit(xtrain, ytrain)
knn_predict = knn_scv.predict(xtest)
knn_results = cross_val_score(knn_scv, xsa, ysa, cv = 10)

print('score (train):', '%.4f' % knn_scv.score(xtrain, ytrain))
print('score (test):','%.4f' % knn_scv.score(xtest, ytest))
print('cross evaluation:', '%.4f' % np.mean(knn_results))
print("classification report:", classification_report(ytest, knn_predict))
print('confusion matrix:', confusion_matrix(ytest, knn_predict ))

# score (train): 0.7532
# score (test): 0.7479
# cross evaluation: 0.5652
# classification report:               precision    recall  f1-score   support

#            0       0.77      0.96      0.85      3244
#            1       0.30      0.05      0.09      1001

#     accuracy                           0.75      4245
#    macro avg       0.53      0.51      0.47      4245
# weighted avg       0.66      0.75      0.67      4245

# confusion matrix: [[3125  119]
#  [ 951   50]]
# %% [markdown]
## Decision Tree
dt = DecisionTreeClassifier(criterion= 'entropy', max_depth= 10, random_state= 1)
fit5 = dt.fit(xtrain, ytrain)
dt_predict = dt.predict(xtest)
dt_results = cross_val_score(dt, xsa, ysa, cv = 10)

print('score (train):', '%.4f' % dt.score(xtrain, ytrain))
print('score (test):','%.4f' % dt.score(xtest, ytest))
print('cross evaluation:', '%.4f' % np.mean(dt_results))
print("classification report:", classification_report(ytest, dt_predict))
print('confusion matrix:', confusion_matrix(ytest, dt_predict ))


# dt = DecisionTreeClassifier(criterion= 'entropy', max_depth= 10, random_state= 1)...
# score (train): 0.7720
# score (test): 0.7637
# cross evaluation: 0.6049
# classification report:               precision    recall  f1-score   support

#            0       0.76      1.00      0.87      3244
#            1       0.33      0.00      0.00      1001

#     accuracy                           0.76      4245
#    macro avg       0.55      0.50      0.43      4245
# weighted avg       0.66      0.76      0.66      4245

# confusion matrix: [[3240    4]
#  [ 999    2]]

# %% [markdown]
## Kmeans
#from sklearn.cluster import KMeans
#km_x = KMeans( n_clusters= 3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
#y_km = km_x.fit_predict(x)
#index1 = 2
#index2 = 3
#plt.scatter( x[y_km==0].iloc[:,index1], x[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )
#plt.scatter( x[y_km==1].iloc[:,index1], x[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )
#plt.scatter( x[y_km==2].iloc[:,index1], x[y_km==2].iloc[:,index2], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3' )
# plot the centroids
#plt.scatter( km_x.cluster_centers_[:, index1], km_x.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
#plt.legend(scatterpoints=1)
#plt.xlabel(str(index1) + " : " + x.columns[index1])
#plt.ylabel(str(index2) + " : " + x.columns[index2])
#plt.grid()
#plt.show()
# %% [markdown]
## SVC
svc = SVC() 
fit6 = svc.fit(xtrain,ytrain)
svc_predict = svc.predict(xtest)
svc_results = cross_val_score(svc, xsa, ysa, cv = 10)

print(f'svc train score:  {svc.score(xtrain,ytrain)}')
print(f'svc test score:  {svc.score(xtest,ytest)}')
print('cross evaluation:', '%.4f' % np.mean(svc_results))
print(classification_report(ytest, svc.predict(xtest)))
print(confusion_matrix(ytest, svc.predict(xtest)))

# svc train score:  0.7716349929323072
# svc test score:  0.7641931684334511
# cross evaluation: 0.7193
#               precision    recall  f1-score   support

#            0       0.76      1.00      0.87      3244
#            1       0.00      0.00      0.00      1001

#     accuracy                           0.76      4245
#    macro avg       0.38      0.50      0.43      4245
# weighted avg       0.58      0.76      0.66      4245

# [[3244    0]
#  [1001    0]]
#%%
## SVC variation
svc2 = SVC(gamma= 'auto')
fit7 = svc2.fit(xtrain,ytrain)
svc2_predict = svc2.predict(xtest)
svc2_results = cross_val_score(svc2, xsa, ysa, cv = 10)
print(f'svc auto train score:  {svc2.score(xtrain,ytrain)}')
print(f'svc auto test score:  {svc2.score(xtest,ytest)}')
print('cross evaluation:', '%.4f' % np.mean(svc2_results))
print(confusion_matrix(ytest, svc2.predict(xtest)))
print(classification_report(ytest, svc2.predict(xtest)))

# svc auto train score:  0.7719491126119051
# svc auto test score:  0.7637220259128387
# cross evaluation: 0.7221
# [[3240    4]
#  [ 999    2]]
#               precision    recall  f1-score   support

#            0       0.76      1.00      0.87      3244
#            1       0.33      0.00      0.00      1001

#     accuracy                           0.76      4245
#    macro avg       0.55      0.50      0.43      4245
# weighted avg       0.66      0.76      0.66      4245


# %% [markdown]
## LinearSVC
linsvc = LinearSVC()
fit8 = linsvc.fit(xtrain,ytrain)
linsvc_predict = linsvc.predict(xtest)
linsvc_results = cross_val_score(linsvc, xsa, ysa, cv = 10)
print(f'Linear SVC scale train score:  {linsvc.score(xtrain,ytrain)}')
print(f'Linear SVC test score:  {linsvc.score(xtest,ytest)}')
print(confusion_matrix(ytest, linsvc.predict(xtest)))
print(classification_report(ytest, linsvc.predict(xtest)))


# Linear SVC scale train score:  0.7716349929323072
# Linear SVC test score:  0.7641931684334511
# [[3244    0]
#  [1001    0]]
#               precision    recall  f1-score   support

#            0       0.76      1.00      0.87      3244
#            1       0.00      0.00      0.00      1001

#     accuracy                           0.76      4245
#    macro avg       0.38      0.50      0.43      4245
# weighted avg       0.58      0.76      0.66      4245


