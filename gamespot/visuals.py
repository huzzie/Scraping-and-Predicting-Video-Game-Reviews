#%%
import seaborn as sns
sns.set_theme()
sns.set(rc={'figure.figsize':(11.7,8.27)})
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('gamespot_reviews.csv')
df.info()

#%%
sns.histplot(df, x='score', bins=20)

#%%
# pd_df = df.sort_values(['score']).reset_index(drop=True)
# print (pd_df)
#%%
sns.countplot(y='genre', data=df, order = df['genre'].value_counts().index)

#%%
sns.boxplot(data=df, x='score', y='genre', order = df['genre'].value_counts().index)

#%%
sns.countplot(y='platform', data=df, order = df['platform'].value_counts().index)

#%%
sns.boxplot(data=df, x='score', y='platform', order = df['platform'].value_counts().index)

#%%
# Scores distribution for all platforms and genres
# sns.displot(df, x="score", col="platform", col_wrap=4, multiple="dodge")

#%%
# def scoreDist(platformList):
#     for platform in platformList:
#         subset = df [df['platform'] == platform]
#         sns.displot(subset, x='score', col='genre', col_wrap=5, hue='genre', multiple='dodge')
#     return

#%%
# platform_list = ['3DS', 'Nintendo Switch', 'PC', 'PlayStation 3', 'PlayStation 4', 'PlayStation Vita', 'Wii', 'Wii U', 'Xbox 360', 'Xbox One']
# scoreDist(platform_list)

# %%
