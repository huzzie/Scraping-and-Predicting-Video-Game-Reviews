# %%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# %%
df = pd.read_excel(r'E:\Chrome\project (1)\Opencritic\OC.xlsx')
print(df.info())
print(df['platform'].unique())
df = df.dropna()

# %%
plt.hist(df.score, bins=20)
plt.show()


# %%
def scoreDist(platformList):
    for platform in platformList:
        subset = df[df['platform'] == platform]
        plt.hist(subset.score)
        plt.show()
    return


# %%
platform_list = df['platform'].unique().tolist()
scoreDist(platform_list)

# %%
df3DS = df[df['platform'] == 'Nintendo Switch']
x = pd.DataFrame(df3DS.groupby(['genre'])['score'].mean())
# print(x)
# print(x.dtypes)
x.plot.barh(color='k', alpha=0.7)
plt.show()

# %%
dfPC = df[df['platform'] == 'PC']
x = pd.DataFrame(dfPC.groupby(['genre'])['score'].mean())
# print(x)
# print(x.dtypes)
x.plot.barh(color='k', alpha=0.7)
plt.show()

# %%
sns.boxplot(x='score', y='genre', orient='h', data=df3DS)

# %%
data = pd.read_excel(r'E:\Chrome\project (1)\Opencritic\OC.xlsx')
col = ['platform', 'genre']
df = data[col].astype('category')
df_code = pd.DataFrame({col: df[col].cat.codes for col in df}, index=df.index)
data = pd.concat([data[['score']], df_code], axis=1)
data = data.dropna()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(data, cmap='Blues')
plt.show()

# %%
