#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#%%
os.getcwd()

#%%
gamespot_df = pd.read_csv('gamespot_reviews.csv')
metacritic_df = pd.read_csv('../metacritic/all_metacritics_reviews_updated.csv')
opencritic_df = pd.read_csv('../opencritic/OC.csv')

# Add site to each dataframe
gamespot_df['site'] = 'gamespot'
metacritic_df['site'] = 'metacritic'
opencritic_df['site'] = 'opencritic'

reviews = [gamespot_df, metacritic_df, opencritic_df]

#%%
all_reviews = gamespot_df.append([metacritic_df, opencritic_df])
all_reviews.drop(['Unnamed: 0', 'Unnamed: 0.1', 'nplatform', 'ngenre'], inplace=True, axis=1)
print(all_reviews.head())
print(all_reviews.shape)
print(all_reviews.site.value_counts())

all_reviews.to_csv('all_reviews.csv')


#%%

####################################
# for later use to clean on my own #
####################################

#%%

def removeData(df, column, data):
    df = df[df[column] != data]
    return df

bad_data = ['PlayStation VR', 'Oculus Rift', 'PlayStation 5', 'Xbox Series X/S', 'HTC Vive', 'Art', 'Art|Adventure', 'Art|Puzzle', 'Creation', 'Creation|Platformer', 'Horror','Interactive Story','Interactive Story|Adventure', 'Music',]

for i in bad_data:
    removeData(all_reviews, 'genre', i)

all_reviews = all_reviews[all_reviews.platform != 'PlayStation VR']
all_reviews = all_reviews[all_reviews.platform != 'Oculus Rift']
all_reviews = all_reviews[all_reviews.platform != 'PlayStation 5']
all_reviews = all_reviews[all_reviews.platform != 'Xbox Series X/S']
all_reviews = all_reviews[all_reviews.platform != 'HTC Vive']
all_reviews = all_reviews[all_reviews.genre != 'Art']
all_reviews = all_reviews[all_reviews.genre != 'Art|Adventure']
all_reviews = all_reviews[all_reviews.genre != 'Art|Adventure']

all_reviews.platform.replace({'3DS': 'Nintendo 3DS', 'Wii-U': 'Wii U', 'Switch': 'Nintendo Switch'}, inplace=True)

# Replace all action genres with just action
all_reviews.genre.replace({'Action|Adventure': 'Action', 'Action|Adventure|Platformer': 'Action','Action|Adventure|RPG': 'Action','Action|First-Person Shooter': 'Action','Action|First-Person Shooter|Vehicle Combat': 'Action','Action|Platformer': 'Action','Action|RPG': 'Action ','Action|RPG|Strategy': 'Action','Action|Real-Time Strategy': 'Action','Action|Roguelike': 'Action','Action|Sports': 'Action','Action|Strategy': 'Action'}, inplace=True)

all_reviews.genre.replace({'Adventure|First-Person Shooter': 'Adventure','Adventure|Platformer': 'Adventure','Adventure|RPG': 'Adventure','Adventure|RPG|Action': 'Adventure','Adventure|RPG|Turn-Based Strategy': 'Adventure','Adventure|Simulation': 'Adventure','Adventure|Third-Person Shooter': 'Adventure'}, inplace=True)

all_reviews.genre.replace({}, inplace=True)
all_reviews.genre.replace({}, inplace=True)
all_reviews.genre.replace({}, inplace=True)
all_reviews.genre.replace({}, inplace=True)
all_reviews.genre.replace({}, inplace=True)
all_reviews.genre.replace({}, inplace=True)
all_reviews.genre.replace({}, inplace=True)
all_reviews.genre.replace({}, inplace=True)
all_reviews.genre.replace({}, inplace=True)
all_reviews.genre.replace({}, inplace=True)
all_reviews.genre.replace({}, inplace=True)


#%%
sns.set()
sns.countplot(y='genre', data=all_reviews, hue='site', order = all_reviews['genre'].value_counts().index)

# %%
