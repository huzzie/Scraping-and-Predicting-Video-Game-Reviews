# %%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# %%
df = pd.read_csv(r'E:\Chrome\project (1)\Opencritic\OC.csv', encoding="gbk")
print(df.info())
print(df['platform'].unique())
df = df.dropna()

columns = [df.platform, df.genre, df.score]
df_m = pd.DataFrame(data=columns)
print(df_m.head())

# %%
x = df['score'].values
sns.distplot(x)
plt.show()

plt.hist(x, color='red', histtype='bar', rwidth=0.97,bins=256)
plt.show()


# %%
def scoreDist(platformList):
    for platform in platformList:
        subset = df[df['platform'] == platform]
        sns.distplot(subset['score'].values)
    return


# %%
platform_list = df['platform'].unique().tolist()[:10]
scoreDist(platform_list)
plt.show()

# %%
sns.distplot(df['score'].values)
plt.show()
# %%
sns.boxplot(x='score', y='genre', data=df[df['genre'].isin(df['genre'].unique().tolist())])
plt.show()
# %%
sns.boxplot(x='score', y='platform', data=df[df['platform'].isin(df['platform'].unique().tolist()[:10])])
plt.show()

print(pd.value_counts(df['platform']))
x = pd.value_counts(df['genre'])
x.to_csv(r"E:\Chrome\project (1)\Opencritic\123.csv")
