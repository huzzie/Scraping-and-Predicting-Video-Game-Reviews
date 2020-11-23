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


# %%
os.chdir('/Users/hyunjaecho/Documents/GitHub/Team-Amazing')

#%%
genres = ['games', 'movies', 'tv', 'music']
scores_by = ['metascore', 'userscore']
#%%
#game_sites = 'https://www.metacritic.com/browse/' + genres + scores_by + '/year/all/filtered'

#%%
driver=webdriver.Chrome(executable_path=r"/Users/HyunjaeCho/Documents/Fall2020/DataMining/chromedriver")
games_site = 'https://www.metacritic.com/browse/games/score/metascore/year/all/filtered'
driver.get(games_site)
html = driver.page_source
bsobj = BeautifulSoup(html, 'html.parser')
platform = [i.text for i in bsobj.find_all('div', {'class': 'platform'})]
platform = platform[1:]
title = [i.text for i in bsobj.find_all('a', {'class': 'title'})]
critic_grade = [row.text for row in bsobj.find_all('div', {'class': 'metascore_w large game positive'})]
critic_grade = critic_grade[::2] #since it extracts two scores from one 
df_game = pd.DataFrame({'platform': platform, 'title': title, 'score': critic_grade, 'genre': 'games'})

#%%
driver=webdriver.Chrome(executable_path=r"/Users/HyunjaeCho/Documents/Fall2020/DataMining/chromedriver")
games_site = 'https://www.metacritic.com/browse/movies/score/metascore/year/filtered'
driver.get(games_site)
html = driver.page_source
bsobj = BeautifulSoup(html, 'html.parser')
# no platform
title = [i.text for i in bsobj.find_all('a', {'class': 'title'})]
critic_grade = [row.text for row in bsobj.find_all('div', {'class': 'metascore_w large movie positive'})]
critic_grade = critic_grade[::2] #since it extracts two scores from one 
df_movie = pd.DataFrame({'platform': 'no platform', 'title': title, 'score': critic_grade, 'genre': 'movie'})

#%%
driver=webdriver.Chrome(executable_path=r"/Users/HyunjaeCho/Documents/Fall2020/DataMining/chromedriver")
site = 'https://www.metacritic.com/browse/tv/score/metascore/year/filtered?sort=desc&view=detailed'
driver.get(site)
html = driver.page_source
bsobj = BeautifulSoup(html, 'html.parser')
# no platform
title = [i.text for i in bsobj.find_all('a', {'class': 'title'})]
critic_grade = [row.text for row in bsobj.find_all('div', {'class': 'metascore_w large season positive'})]
critic_grade = critic_grade[::2] #since it extracts two scores from one 
df_tv = pd.DataFrame({'platform': 'no platform', 'title': title, 'score': critic_grade, 'genre': 'music'})


#%%
driver=webdriver.Chrome(executable_path=r"/Users/HyunjaeCho/Documents/Fall2020/DataMining/chromedriver")
site = 'https://www.metacritic.com/browse/albums/score/metascore/year/filtered'
driver.get(site)
html = driver.page_source
bsobj = BeautifulSoup(html, 'html.parser')
# no platform
title = [i.text for i in bsobj.find_all('a', {'class': 'title'})]
critic_grade = [row.text for row in bsobj.find_all('div', {'class': 'metascore_w large release positive'})]
critic_grade = critic_grade[::2] #since it extracts two scores from one 
df_music = pd.DataFrame({'platform': 'no platform', 'title': title, 'score': critic_grade, 'genre': 'tv'})


#%%
total = [df_game, df_music, df_movie, df_tv]
total = pd.concat(total)

#%%
total.to_csv('metacritic_reviews.csv', encoding= 'utf-8')



#%%
driver=webdriver.Chrome(executable_path=r"/Users/HyunjaeCho/Documents/Fall2020/DataMining/chromedriver")
games_site = 'https://www.metacritic.com/browse/games/score/metascore/year/all/filtered'
driver.get(games_site)

df = pd.DataFrame({'platform': platform, 'title': title, 'score': critic_grade, 'genre': 'games'})

### Game scores by users
try:
    driver.find_element_by_xpath("//div[@class='mcmenu dropdown style1 sort']").click()
    driver.find_element_by_xpath("//*[@id='main_content']/div[1]/div[2]/div/div[1]/div/div[1]/div[2]/div[1]/ul/li[2]/a").click()
    driver.implicitly_wait(5)
except:
    print('working')
driver.implicitly_wait(5)

while True:
    try:
        html = driver.page_source
        bsobj = BeautifulSoup(html, 'html.parser')
        platform = [i.text for i in bsobj.find_all('div', {'class': 'platform'})]
        platform = platform[1:]
        title = [i.text for i in bsobj.find_all('a', {'class': 'title'})]
        critic_grade = [row.text for row in bsobj.find_all('div', {'class': 'metascore_w large game positive'})]
        critic_grade = critic_grade[::2] #since it extracts two scores from one 
    except:
        print('workinggg')
driver.quit()

df = pd.DataFrame({'platform': platform, 'title': title, 'score': critic_grade, 'genre': 'games'})
print(df)


#%%


# %%
