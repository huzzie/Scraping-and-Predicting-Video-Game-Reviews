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
print(total_df.head(10))

#%%
## describe the number of genre and platform
total_df['genre'].value_counts()
total_df['platform'].value_counts()
#%%
# total df plot
sns.histplot(total_df, x = 'score', bins = 20)


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
# score (train): 0.0004
print('score (test):','%.4f' % full_split1.score(xtest, ytest))
# score (test): 0.0008
print('intercept:','%.4f' % full_split1.intercept_)
# intercept: 69.2962
print('coef_:', full_split1.coef_)
# coef_: [ 0.04338489 -0.02363915]
print('cross evaluation:', cross_val_score(fit, x, y, cv = 10))
#cross evaluation: [-3.69393243e-01 -3.61725768e-02 -2.67365799e+01 -5.85679124e+00
#-2.31284593e+00 -1.55551828e+00 -1.12322716e-02 -1.81602225e-02
#-1.13455910e-01 -2.30535166e-01]

# %% [markdown]
## Logistic Model

#%%
logit = LogisticRegression()
fit2 = logit.fit(xtrain, ytrain)
print('score (train):', '%.4f' % logit.score(xtrain, ytrain))
# score (train): 0.0372
print('score (train):', '%.4f' % logit.score(xtest, ytest))
# score (train): 0.0356
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
# score (train): 0.7604
print('score (test):','%.4f' % scorelogit.score(xtest, ytest))
# score (test): 0.7611
print('intercept:','%.4f' % scorelogit.intercept_)
# intercept: -1.1059
print('coef_:', scorelogit.coef_)
# coef_:[[-0.0138512   0.02059239]]
print('cross evaluation:', cross_val_score(fit3, x, y, cv = 10))
#cross evaluation: [0.7604533  0.7604533  0.7604533  0.7604533  0.7604533  0.7604533
# 0.76084408 0.76075059 0.76075059 0.76075059]
print("classification report:", classification_report(ytest, y_predit))

#               precision    recall  f1-score   support
#
#           0       0.76      1.00      0.86      4869
#           1       0.00      0.00      0.00      1528
#
#    accuracy                           0.76      6397
#   macro avg       0.38      0.50      0.43      6397
#weighted avg       0.58      0.76      0.66      6397
print('confusion matrix:', confusion_matrix(ytest, y_predit ))
# [[4869 0]
#  [1528 0]]
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
# score (train): 0.7520
print('score (test):','%.4f' % knn_scv.score(xtest, ytest))
# score (test): 0.7480
print('cross evaluation:', '%.4f' % np.mean(knn_results))
# cross evaluation: 0.6213
print("classification report:", classification_report(ytest, knn_predict))

#                 precision    recall  f1-score   support
#
#           0       0.77      0.96      0.85      4869
#           1       0.34      0.06      0.10      1528
#    accuracy                           0.75      6397
#   macro avg       0.55      0.51      0.48      6397
#weighted avg       0.66      0.75      0.67      6397
print('confusion matrix:', confusion_matrix(ytest, knn_predict ))
#[[4696  173]
# [1439   89]]
# %% [markdown]
## Decision Tree
dt = DecisionTreeClassifier(criterion= 'entropy', max_depth= 10, random_state= 1)
fit5 = dt.fit(xtrain, ytrain)
dt_predict = dt.predict(xtest)
dt_results = cross_val_score(dt, xsa, ysa, cv = 10)

print('score (train):', '%.4f' % dt.score(xtrain, ytrain))
#score (train): 0.7619
print('score (test):','%.4f' % dt.score(xtest, ytest))
#score (test): 0.7611
print('cross evaluation:', '%.4f' % np.mean(dt_results))
# cross evaluation: 0.6512
print("classification report:", classification_report(ytest, dt_predict))
#classification report:               precision    recall  f1-score   support
#           0       0.76      1.00      0.86      4869
#           1       0.50      0.01      0.01      1528
#    accuracy                           0.76      6397
#   macro avg       0.63      0.50      0.44      6397
#weighted avg       0.70      0.76      0.66      6397
print('confusion matrix:', confusion_matrix(ytest, dt_predict ))
#confusion matrix: [[4858   11]
#                  [1517   11]]

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
# svc train score:  0.7603
print(f'svc test score:  {svc.score(xtest,ytest)}')
# svc test score:  0.7611
print('cross evaluation:', '%.4f' % np.mean(svc_results))
# cross evaluation: 0.7606
print(classification_report(ytest, svc.predict(xtest)))
#               precision    recall  f1-score   support

#            0       0.76      1.00      0.86      4869
#            1       0.00      0.00      0.00      1528

#     accuracy                           0.76      6397
#    macro avg       0.38      0.50      0.43      6397
# weighted avg       0.58      0.76      0.66      6397
print(confusion_matrix(ytest, svc.predict(xtest)))
# [[4869    0]
#  [1528    0]]
#%%
## SVC variation
svc2 = SVC(gamma= 'auto')
fit7 = svc2.fit(xtrain,ytrain)
svc2_predict = svc2.predict(xtest)
svc2_results = cross_val_score(svc2, xsa, ysa, cv = 10)
print(f'svc auto train score:  {svc2.score(xtrain,ytrain)}')
# svc auto train score:  0.7611
print(f'svc auto test score:  {svc2.score(xtest,ytest)}')
#svc auto test score:  0.7609
print('cross evaluation:', '%.4f' % np.mean(svc2_results))
# cross evaluation: 0.7606
print(confusion_matrix(ytest, svc2.predict(xtest)))
# [[4864    5]
#  [1524    4]]
print(classification_report(ytest, svc2.predict(xtest)))
#              precision    recall  f1-score   support

#            0       0.76      1.00      0.86      4869
#            1       0.44      0.00      0.01      1528

#     accuracy                           0.76      6397
#    macro avg       0.60      0.50      0.43      6397
# weighted avg       0.69      0.76      0.66      6397

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

# linsvc = LinearSVC()...
# Linear SVC scale train score:  0.7603960396039604
# Linear SVC test score:  0.7611380334531812
# [[4869    0]
#  [1528    0]]
#               precision    recall  f1-score   support

#            0       0.76      1.00      0.86      4869
#            1       0.00      0.00      0.00      1528

#     accuracy                           0.76      6397
#    macro avg       0.38      0.50      0.43      6397
# weighted avg       0.58      0.76      0.66      6397
