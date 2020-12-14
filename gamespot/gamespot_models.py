#%%
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn import neighbors 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%%
df_gamespot = pd.read_csv('gamespot_reviews.csv')

#%%[markdown]
# # Linear Regression Model

#%%
# Turn categorical to coded numeric
platform_dict = {'3DS': 0, 'Nintendo Switch': 1, 'PC': 2, 'PlayStation 3': 3, 'PlayStation 4': 4, 'PlayStation Vita': 4, 'Wii': 5, 'Wii U': 6, 'Xbox 360': 7, 'Xbox One': 8}

genre_dict = {'Action': 0, 'Adventure': 1, 'Fighting': 2, 'First-Person': 3, 'Flight': 4, 'Party/Minigame': 5, 'Platformer': 6, 'Puzzle': 7, 'Driving/Racing': 8, 'Real-Time': 9, 'Role-Playing': 10, 'Simulation': 11, 'Sports': 12, 'Strategy': 13, 'Third-Person': 14, 'Turn-Based': 15, 'Wrestling': 16}

df_gamespot['platform_coded'] = df_gamespot['platform'].map(platform_dict)
df_gamespot['genre_coded'] = df_gamespot['genre'].map(genre_dict)

# print(df_gamespot.head())

#%%
# Set dependent and independent variables
x = df_gamespot[['platform_coded', 'genre_coded']]
y = df_gamespot['score']

# %%
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state=2020)

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

# The accuracy of the train set: 0.002768614303998107

# The accuracy of the test set: 0.0037527955246577926

# Intercept: 6.802165115328733

# Coefficient: [0.00218857 0.01566698]

# Cross evaluation
# [-0.00668987 -0.01715508 -0.00512538 -0.00067753 -0.00358889 -0.00350473
#  -0.01716809 -0.01129756 -0.00277276 -0.02214194]

# Cross evaluation mean: -0.009012182813006064

#%%
# Check for multicollinearity in factors

# Add intercept term to x dataframe
x['Intercept'] = lm.intercept_

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = x.columns
vif["VIF"] = [ variance_inflation_factor(x.values, i) for i in range(x.shape[1]) ]
print(vif) # No issues with multicollinearity

#%%[markdown]
# # Logistic model
# Can we predict if a game is going to get a score of 8 or better?

#%%
# Update dataframe to code scores as 8 or better

# create a list of conditions for the score cutoff
conditions = [
    (df_gamespot['score'] < 8),
    (df_gamespot['score'] >= 8)
    ]

# create a list of the values we want to assign for each condition
values = [0, 1]

# create a new column and use np.select to assign values to it using our lists as arguments
df_gamespot['score8'] = np.select(conditions, values)

# print(df_gamespot.score8.value_counts())

#%%
# Set dependent and independent variables
xcf = df_gamespot[['platform_coded', 'genre_coded']]
ycf = df_gamespot['score8']

xtraincf, xtestcf, ytraincf, ytestcf = train_test_split(xcf, ycf, test_size = 0.2, random_state=2020)

scoreLogit = LogisticRegression()
scoreLogitFit = scoreLogit.fit(xtraincf, ytraincf)
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

# The accuracy of the train set: 0.6493332054111101

# The accuracy of the test set: 0.6607828089025326

# Cross evaluation accuracies:
# [0.65157329 0.65157329 0.65157329 0.65157329 0.65157329 0.65157329
#  0.65157329 0.65157329 0.65157329 0.65207373]

# Cross evaluation mean: 0.6516233364338234

# Predicted probabilities of train
# [[0.65294661 0.34705339]
#  [0.63621819 0.36378181]
#  [0.63856304 0.36143696]
#  ...
#  [0.65515314 0.34484686]
#  [0.64405536 0.35594464]
#  [0.65294661 0.34705339]]

# Predicted probabilities of test
# [[0.64537756 0.35462244]
#  [0.64537756 0.35462244]
#  [0.65735299 0.34264701]
#  ...
#  [0.64742655 0.35257345]
#  [0.63529667 0.36470333]
#  [0.65335279 0.34664721]]

# Classification report:
#                precision    recall  f1-score   support

#            0       0.66      1.00      0.80      1722
#            1       0.00      0.00      0.00       884

#     accuracy                           0.66      2606
#    macro avg       0.33      0.50      0.40      2606
# weighted avg       0.44      0.66      0.53      2606

# Confusion matrix:
#  [[1722    0]
#  [ 884    0]]

#%%
# ROC-AUC Curve

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

# ROC-AUC Curve...
# Poor model
# No Skill: ROC AUC=0.500
# Logistic: ROC AUC=0.547
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
# %% [markdown]
# # KNN model

#%%
knn = KNeighborsClassifier(n_neighbors=7)
knnFit = knn.fit(xtraincf,ytraincf)
knnPredict = knn.predict(xtestcf)
knnTrain = knn.score(xtraincf,ytraincf)
knnTest = knn.score(xtestcf,ytestcf)
knn_cv = cross_val_score(knnFit, xcf, ycf, cv=10, scoring='accuracy')

print(f'Accuracy of KNN train model: {knnTrain}\n')
print(f'Accuracy of KNN test model: {knnTest}\n')
print(f'Cross evaluation accuracies:\n{knn_cv}\n\nCross evaluation mean: {np.mean(knn_cv)}') # [0.3637759  0.65157329 0.45817345 0.29470453 0.44973139 0.55564083 0.50652341 0.55871067 0.38142748 0.48463902]
# mean: 0.47048999531979263
print(f'Classification report:\n {classification_report(ytestcf, knnPredict)}')
print(f'Confusion matrix:\n {confusion_matrix(ytestcf, knnPredict)}'),

# Accuracy of KNN train model: 0.6316799385973328

# Accuracy of KNN test model: 0.6327705295471988

# Cross evaluation accuracies:
# [0.3637759  0.65157329 0.45817345 0.29470453 0.44973139 0.55564083
#  0.50652341 0.55871067 0.38142748 0.48463902]

# Cross evaluation mean: 0.47048999531979263
# Classification report:
#                precision    recall  f1-score   support

#            0       0.66      0.91      0.77      1722
#            1       0.35      0.10      0.15       884

#     accuracy                           0.63      2606
#    macro avg       0.51      0.50      0.46      2606
# weighted avg       0.56      0.63      0.56      2606

# Confusion matrix:
#  [[1562  160]
#  [ 797   87]]

#%%
def knn_comparison(data, k):
    x = data[['platform_coded', 'genre_coded']].values
    y = data['score8'].astype(int).values
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(x, y)
    # Plotting decision region
    plot_decision_regions(x, y, clf=clf, legend=2)
    # Adding axes annotations
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Knn with K='+ str(k))
    plt.show()

for i in [1,5,7,20,30,40,80]:
    knn_comparison(df_gamespot, i)
# %% [markdown]
# Decision Tree

#%%
# Instantiate dtree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)
tree_fit = tree.fit(xtraincf,ytraincf)
tree_train = tree.score(xtraincf, ytraincf)
tree_test = tree.score(xtestcf, ytestcf)
tree_pred = tree.predict(xtestcf)
tree_cv = cross_val_score(tree_fit, xcf, ycf, cv=10, scoring='accuracy')

print(f'Decision Tree train accuracy score: {tree_train}\n')
# Same as next line: 
# print(f'Decision Tree accuracy: {accuracy_score(ytestcf, tree_pred)}\n')
print(f'Decision Tree test accuracy score: {tree_test}\n')
print(confusion_matrix(ytestcf, tree_pred))
print(classification_report(ytestcf, tree_pred))
print(f'Cross evaluation accuracies:\n{tree_cv}\n\nCross evaluation mean: {np.mean(tree_cv)}')

# Decision Tree train accuracy score: 0.6569125971409383

# Decision Tree test accuracy score: 0.6588641596316194

# [[1658   64]
#  [ 825   59]]
#               precision    recall  f1-score   support

#            0       0.67      0.96      0.79      1722
#            1       0.48      0.07      0.12       884

#     accuracy                           0.66      2606
#    macro avg       0.57      0.51      0.45      2606
# weighted avg       0.60      0.66      0.56      2606
#
# Cross evaluation accuracies:
# [0.65157329 0.65157329 0.65157329 0.37989256 0.39984651 0.62624712
#  0.52033768 0.50345357 0.61627015 0.37403994]

# Cross evaluation mean: 0.5374807398264433

#%%
# filler_feature_values is used when you have more than 2 predictors, then 
# you need to specify the ones not shown in the 2-D plot. For us, 
# the rank is at poition 2, and the value can be 1, 2, 3, or 4.
# also need to specify the filler_feature_ranges for +/-, otherwise only data points with that feature value will be shown.

# And the decision tree result
plot_decision_regions(xtestcf.values, ytestcf.values, clf=tree, legend=3, filler_feature_values={2:1} , filler_feature_ranges={2: 3} )
plt.xlabel(xtestcf.columns[0])
plt.ylabel(xtestcf.columns[1])
plt.title(tree.__class__.__name__)
plt.legend(loc="upper left")
plt.show()

# %%[markdown]
# # SVC

#%%
svc = SVC(gamma='auto')
svcFit = svc.fit(xtraincf,ytraincf)
svc_cv_acc = cross_val_score(svc, xcf, ycf, cv= 10, scoring='accuracy')
print(f'SVC train score:  {svc.score(xtraincf,ytraincf)}')
print(f'SVC test score:  {svc.score(xtestcf,ytestcf)}')
print(f'SVC CV accuracy score: {svc_cv_acc}\n\n SVC CV mean accuracy: {np.mean(svc_cv_acc)}\n')
print(confusion_matrix(ytestcf, svc.predict(xtestcf)))
print(classification_report(ytestcf, svc.predict(xtestcf)))

# SVC train score:  0.6548978221241485
# SVC test score:  0.6638526477359938
# SVC CV accuracy score: [0.64525407 0.64908917 0.64237776 0.65163148 0.64779271 0.64395393
#  0.65259117 0.65355086 0.64587332 0.64299424]

#  SVC CV mean accuracy: 0.6475108713054584

# [[1676   46]
#  [ 830   54]]
#               precision    recall  f1-score   support

#            0       0.67      0.97      0.79      1722
#            1       0.54      0.06      0.11       884

#     accuracy                           0.66      2606
#    macro avg       0.60      0.52      0.45      2606
# weighted avg       0.63      0.66      0.56      2606

#%%[markdown]
# # Linear SVC

#%%
# Get 0 for precision and 0s in confusion matrix
linearSVC = LinearSVC()
linearSVCFit = linearSVC.fit(xtraincf,ytraincf)
linearSVC_cv_acc = cross_val_score(linearSVC, xtraincf, ytraincf, cv= 10, scoring='accuracy')
print(f'linearSVC train score:  {linearSVC.score(xtraincf,ytraincf)}')
print(f'linearSVC test score:  {linearSVC.score(xtestcf,ytestcf)}')
print(f'linearSVC CV accuracy score:  {linearSVC_cv_acc}\n\nCross evaluation mean: {np.mean(linearSVC_cv_acc)}')
print(f'Confusion matrix:\n {confusion_matrix(ytestcf, linearSVC.predict(xtestcf))}')
print(f'Classification report:\n {classification_report(ytestcf, linearSVC.predict(xtestcf))}')

# linearSVC train score:  0.6493332054111101
# linearSVC test score:  0.6607828089025326
# linearSVC CV accuracy score:  [0.64908917 0.64908917 0.64908917 0.64971209 0.64971209 0.64971209
#  0.64971209 0.64971209 0.6487524  0.6487524 ]

# Cross evaluation mean: 0.649333275672015
# Confusion matrix:
#  [[1722    0]
#  [ 884    0]]
# Classification report:
#                precision    recall  f1-score   support

#            0       0.66      1.00      0.80      1722
#            1       0.00      0.00      0.00       884

#     accuracy                           0.66      2606
#    macro avg       0.33      0.50      0.40      2606
# weighted avg       0.44      0.66      0.53      2606

# %%
# Plot classifiers
# modified from datacamp

def plot_classifier(X, y, clf, ax=None, ticks=False, proba=False, lims=None): # assumes classifier "clf" is already fit
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    else:
        show = False

    # can abstract some of this into a higher-level function for learners to call
    cs = plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8, proba=proba)
    if proba:
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('probability of red $\Delta$ class', fontsize=20, rotation=270, labelpad=30)
        cbar.ax.tick_params(labelsize=14)
    #ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k', linewidth=1)
    labels = np.unique(y)
    if len(labels) == 2:
        ax.scatter(X0[y==labels[0]], X1[y==labels[0]], cmap=plt.cm.coolwarm, s=60, c='b', marker='o', edgecolors='k')
        ax.scatter(X0[y==labels[1]], X1[y==labels[1]], cmap=plt.cm.coolwarm, s=60, c='r', marker='^', edgecolors='k')
    else:
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k', linewidth=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
#     ax.set_xlabel(data.feature_names[0])
#     ax.set_ylabel(data.feature_names[1])
    if ticks:
        ax.set_xticks(())
        ax.set_yticks(())
#     ax.set_title(title)
    if show:
        plt.show()
    else:
        return ax

def plot_4_classifiers(X, y, clfs):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for clf, ax, title in zip(clfs, sub.flatten(), ("(1)","(2)","(3)","(4)")):
        # clf.fit(X, y)
        plot_classifier(X, y, clf, ax, ticks=True)
        ax.set_title(title)
    plt.show()

def plot_classifiers(X, y, clfs):
    titles = []
    for model in clfs:
      titles.append(model.__class__.__name__)

    # Set-up nx2 grid for plotting.
    nrows = int(len(clfs)/2) # assume len is even for now
    fig, sub = plt.subplots(nrows, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for clf, ax, title in zip(clfs, sub.flatten(), titles):
        # clf.fit(X, y)
        plot_classifier(X, y, clf, ax, ticks=True)
        ax.set_title(title)
    plt.show()
    
#%%
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
# %%
plot_classifiers(xcf.values, ycf.values, [knnFit, svcFit])
# %%
# Plotting logit so the tree_fit shows up
plot_classifiers(xcf.values, ycf.values, [scoreLogitFit, tree_fit])

#%%
