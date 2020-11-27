#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#%%
# Standard quick checks
def dfChkBasics(dframe, valCnt = False): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  # print(f'\n{cnt}: describe(): ')
  # cnt+=1
  # print(dframe.describe())

  # print(f'\n{cnt}: dtypes: ')
  # cnt+=1
  # print(dframe.dtypes)

  try:
    print(f'\n{cnt}: columns: ')
    cnt+=1
    print(dframe.columns)
  except: pass

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1

#%%
df = pd.read_csv('gamespot_reviews.csv')
dfChkBasics(df)

#%%
x = df[['platform', 'genre']]
y = df['score']

# %%
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state=2020)

lm = linear_model.LinearRegression()
lm_fit = lm.fit(xtrain, ytrain)
lm_predict = lm_fit.predict(xtest)
lm_score_train = lm.score(xtrain, ytrain)
lm_score_test = lm.score(xtest, ytest)

print(f'The accuracy of the train set: {lm_score_train}')
print(f'The accuracy of the test set: {lm_score_test}')
print('intercept:', lm.intercept_)
print('coef_:', lm.coef_) 

# %%
