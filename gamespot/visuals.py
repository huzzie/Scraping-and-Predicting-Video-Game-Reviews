#%%
import seaborn as sns
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
sns.displot(df, x="score", col="platform", multiple="dodge")

#%%
df3DS = df[ df['platform'] == '3DS']
sns.displot(df3DS, x="score", row='genre', multiple='dodge')

#%%
dfPC = df[ df['platform'] == 'PC']
sns.displot(dfPC, x="score", row='genre', multiple='dodge')

# %%
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df_m, cmap='Blues')
plt.show()

# %%
