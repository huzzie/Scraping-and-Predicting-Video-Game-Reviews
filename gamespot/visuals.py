#%%
import seaborn as sns
sns.set_theme()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('gamespot_reviews.csv')
df.info()

#%%

columns = [df.platform, df.genre, df.score]
df_m = pd.DataFrame(data=columns)
print(df_m.head())

#%%
sns.histplot(df, x='score', bins=20)

#%%
# Scores distribution for all platforms and genres
sns.displot(df, x="score", col="platform", col_wrap=4, multiple="dodge")

#%%
def scoreDist(platformList):
    for platform in platformList:
        subset = df [df['platform'] == platform]
        sns.displot(subset, x='score', col='genre', col_wrap=5, hue='genre', multiple='dodge')
    return
#%%
platform_list = ['3DS', 'Nintendo Switch', 'PC', 'PlayStation 3', 'PlayStation 4', 'PlayStation Vita', 'Wii', 'Wii U', 'Xbox 360', 'Xbox One']
scoreDist(platform_list)

#%%
df3DS = df[ df['platform'] == '3DS']
sns.displot(df3DS, x="score", col='genre', col_wrap = 5, hue='genre', multiple='dodge', kde=True)

#%%
dfPC = df[ df['platform'] == 'PC']
sns.displot(df3DS, x="score", col='genre', col_wrap = 3, hue='genre', multiple='dodge')

#%%
sns.boxplot(x='score', y='genre', orient='h', data=df3DS)

#%%
dfPC = df[ df['platform'] == 'PC']
sns.displot(dfPC, x="score", row='genre', multiple='dodge')

# %%
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df_m, cmap='Blues')
plt.show()

# %%
