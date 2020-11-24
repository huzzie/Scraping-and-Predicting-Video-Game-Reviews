#%%
import os
from selenium import webdriver
from bs4 import BeautifulSoup
import re
import argparse
import time
import csv
import pandas as pd
from pprint import pprint



#%%
#game_sites = 'https://www.metacritic.com/browse/' + genres + scores_by + '/year/all/filtered'

if not os.path.exists('./metacritic_review'):
    os.makedirs('./metacritic_review')

#%%
def metacritic_review(genre_name):
    os.chdir('/Users/hyunjaecho/Documents/GitHub/Team-Amazing')
    driver=webdriver.Chrome(executable_path=r"/Users/HyunjaeCho/Documents/Fall2020/DataMining/chromedriver")
    total_df = pd.DataFrame([])
    for page in range(0, 6):
        print('current page', page)
        sites = 'https://www.metacritic.com/browse/games/genre/metascore/' + str(genre_name) + '/all?view=detailed' + '&page=' + str(page)
        driver.get(sites)
        print(sites)
        driver.implicitly_wait(10)
        html = driver.page_source
        bsobj = BeautifulSoup(html, 'html.parser')
        platform = [i.text for i in bsobj.find_all('a', {'class': 'title'})]
        #platform = platform[1:]
        #genres = re.findall('.*metascore/(.*?)\/all?.*',sites)
        title = [i.text for i in bsobj.find_all('a', {'class': 'title'})]
        score = [i.text for i in bsobj.find_all('div', {'class': re.compile('metascore_w large game.*')})]
        score = score[::2] #since it extracts two scores at the same time
        genre = [genre_name for i in range(len(score))]
        data = {'platform': platform, 'genre': genre, 'title': title, 'score': score}
        total_df = total_df.append(pd.DataFrame(data))
        total_df.to_csv('metacritic_review_' + str(genre_name) + '.csv', encoding = 'utf-8')

if __name__ == '__main__':
    genres = ['adventure', 'fighting', 'first-person', 'flight', 'party', 'platformer', 'puzzle', 
    'racing', 'real-time', 'role-playing', 'simulation', 'sports', 'strategy', 'third-person',
    'turn-based', 'wargame', 'wrestling']
    for genre_name in zip(genres):
        data = metacritic_review(genre_name)



# %%
import glob
path = r"/Users/HyunjaeCho/Documents/GitHub/Team-Amazing/metacritic_review"
all_files = glob.glob(path + '/*.csv')

li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col= 0, header = 0)
    li.append(df)

total_df = pd.concat(li, axis = 0, ignore_index = True)

#%%
total_df['genre'] = total_df['genre'].str.extract(r"(.*)")
total_df['genre'] = total_df['genre'].str.extract(r'(\w+)')
total_df.to_csv('cleaned_metacritic_reviews.csv', encoding = 'utf-8')
# %%
