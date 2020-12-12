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
lm = linear_model.LinearRegression()
lm_fit = lm.fit(xtrain, ytrain)
lm_predict = lm_fit.predict(xtest)
lm_score_train = lm.score(xtrain, ytrain)
lm_score_test = lm.score(xtest, ytest)
lm_cv = cross_val_score(lm_fit, x, y, cv=10)

print(f'The accuracy of the train set: {lm_score_train}\n')
print(f'The accuracy of the test set: {lm_score_test}\n')
print(f'Intercept: {lm.intercept_}\n')
print(f'Coefficient: {lm.coef_}\n') 
print(f'Cross evaluation\n{lm_cv}\n\nCross evaluation mean: {np.mean(lm_cv)}')


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

# score (train): 0.0079
# score (test): 0.0061
# intercept: 67.6516
# coef_: [ 0.19686723 -0.04670796]
# cross evaluation: [-5.96977521e-01 -2.63904150e-02 -3.65200822e-02  5.94067655e-03
#  -4.04917782e-02 -4.21544804e-02 -1.73931321e-01 -1.97466103e+01
#  -1.74246074e-01 -4.98655688e+00]

# %% [markdown]
## Logistic Model

#%%
logit = LogisticRegression()
fit2 = logit.fit(xtrain, ytrain)
print('score (train):', '%.4f' % logit.score(xtrain, ytrain))
print('score (train):', '%.4f' % logit.score(xtest, ytest))

# score (train): 0.0418
# score (train): 0.0353
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

# score (train): 0.7745
# score (test): 0.7726
# intercept: -1.1976
# coef_: [[-0.00244425 -0.00106777]]
# cross evaluation: [0.77363445 0.77404099 0.77404099 0.77404099 0.77404099 0.77404099
#  0.77404099 0.77404099 0.77404099 0.77404099]
# classification report:               precision    recall  f1-score   support

#            0       0.77      1.00      0.87      3676
#            1       0.00      0.00      0.00      1082

#     accuracy                           0.77      4758
#    macro avg       0.39      0.50      0.44      4758
# weighted avg       0.60      0.77      0.67      4758

# confusion matrix: [[3676    0]
#  [1082    0]]

# %% [markdown]
## Roc-Curve


xcf = total_df[['nplatform', 'ngenre']]
ycf = total_df['score8']

xtraincf, xtestcf, ytraincf, ytestcf = train_test_split(xcf, ycf, test_size = 0.2, random_state=2020)

#%%
scoreLogit = LogisticRegression()
scoreLogitFit = scoreLogit.fit(xtrain, ytrain)
scoreLogit_predict = scoreLogitFit.predict(xtestcf)
scoreLogit_train = scoreLogit.score(xtraincf, ytraincf)
scoreLogit_test = scoreLogit.score(xtestcf, ytestcf)
scoreLogit_cv = cross_val_score(scoreLogitFit, xcf, ycf, cv=10, scoring='accuracy')

print(f'The accuracy of the train set: {scoreLogit_train}\n')
print(f'The accuracy of the test set: {scoreLogit_test}\n')
print(f'Cross evaluation accuracies:\n{scoreLogit_cv}\n\nCross evaluation mean: {np.mean(scoreLogit_cv)}\n')
print(f'Predicted probabilities of train\n{scoreLogit.predict_proba(xtraincf)}\n')
print(f'Predicted probabilities of test\n{scoreLogit.predict_proba(xtestcf)}\n')
print(f'Classification report:\n {classification_report(ytestcf, scoreLogit_predict)}') # Getting 0's across the board for 1 value
print(f'Confusion matrix:\n {confusion_matrix(ytestcf, scoreLogit_predict)}') # Getting 0's across the board for 1 value 

#%%

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(ytestcf))]
# predict probabilities
lr_probs = scoreLogitFit.predict_proba(xtestcf)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(ytestcf, ns_probs)
lr_auc = roc_auc_score(ytestcf, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc)) # 0.500
print('Logistic: ROC AUC=%.3f' % (lr_auc)) # 0.547
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(ytestcf, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(ytestcf, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()



#%%

# %%
precision, recall, thresholds = precision_recall_curve(ytestcf, scoreLogit.predict_proba(xtestcf)[:, 1]) 
# retrieve probability of being 1(in second column of probs_y)
pr_auc = metrics.auc(recall, precision)

plt.title("Logistic Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])

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

# score (train): 0.7678
# score (test): 0.7673
# cross evaluation: 0.6192
# classification report:               precision    recall  f1-score   support

#            0       0.77      0.99      0.87      3676
#            1       0.35      0.03      0.05      1082

#     accuracy                           0.77      4758
#    macro avg       0.56      0.51      0.46      4758
# weighted avg       0.68      0.77      0.68      4758

# confusion matrix: [[3623   53]
#  [1054   28]]
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
# score (train): 0.7748
# score (test): 0.7726
# cross evaluation: 0.7499
# classification report:               precision    recall  f1-score   support

#            0       0.77      1.00      0.87      3676
#            1       0.50      0.00      0.01      1082

#     accuracy                           0.77      4758
#    macro avg       0.64      0.50      0.44      4758
# weighted avg       0.71      0.77      0.67      4758

# confusion matrix: [[3673    3]
#  [1079    3]]

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

# svc train score:  0.7744692776571148
# svc test score:  0.7725935266918873
# cross evaluation: 0.7740
#               precision    recall  f1-score   support

#            0       0.77      1.00      0.87      3676
#            1       0.00      0.00      0.00      1082

#     accuracy                           0.77      4758
#    macro avg       0.39      0.50      0.44      4758
# weighted avg       0.60      0.77      0.67      4758

# [[3676    0]
#  [1082    0]]
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



# ## SVC variation...
# svc auto train score:  0.774679464723604
# svc auto test score:  0.7723833543505675
# cross evaluation: 0.7740
# [[3673    3]
#  [1080    2]]
#               precision    recall  f1-score   support

#            0       0.77      1.00      0.87      3676
#            1       0.40      0.00      0.00      1082

#     accuracy                           0.77      4758
#    macro avg       0.59      0.50      0.44      4758
# weighted avg       0.69      0.77      0.67      4758

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
# Linear SVC scale train score:  0.7744692776571148
# Linear SVC test score:  0.7725935266918873
# [[3676    0]
#  [1082    0]]
#               precision    recall  f1-score   support

#            0       0.77      1.00      0.87      3676
#            1       0.00      0.00      0.00      1082

#     accuracy                           0.77      4758
#    macro avg       0.39      0.50      0.44      4758
# weighted avg       0.60      0.77      0.67      4758


#%%