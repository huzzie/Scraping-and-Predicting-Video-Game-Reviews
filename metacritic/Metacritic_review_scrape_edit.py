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
def hasXpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
        return True
    except:
        return False


#%%

def metacritic_review(genre_name):
    os.chdir('/Users/hyunjaecho/Documents/GitHub/Team-Amazing')
    driver=webdriver.Chrome(executable_path=r"/Users/HyunjaeCho/Documents/Fall2020/DataMining/chromedriver")
    #genre_type = re.findall(r'\w+', str(genre_name))
    genre_type = str(genre_name).replace("['", "")
    genre_type = str(genre_type).replace("']", "")
    genre_type = str(genre_type).replace(",", "")
    genre_type = str(genre_type).replace("('", "")
    genre_type = str(genre_type).replace("')", "")
    site = 'https://www.metacritic.com/browse/games/genre/metascore/' + str(genre_type) + '/all?view=detailed'
    driver.get(site)
    driver.implicitly_wait(10)
    print(site)
    html = driver.page_source
    bsobj = BeautifulSoup(html, 'html.parser')
    last_page = bsobj.find('li', {'class': re.compile('page last_page.*')}).text
    last_page = re.findall('\d+', str(last_page))
    
    titles = []
    platforms = []
    scores = []
    genres = []
    total_df = pd.DataFrame([])
    for page in range(0, int(last_page[0])):
        os.chdir('/Users/hyunjaecho/Documents/GitHub/Team-Amazing')
        driver=webdriver.Chrome(executable_path=r"/Users/HyunjaeCho/Documents/Fall2020/DataMining/chromedriver")
        sites = site + '&page=' + str(page)
        driver.get(sites)
        print(sites)
        driver.implicitly_wait(5)
        html = driver.page_source
        bsobj = BeautifulSoup(html, 'html.parser')
        title = [i.text for i in bsobj.find_all('a', {'class': 'title'})]
        platform = [i.text for i in bsobj.find_all('div', {'class': 'platform'})]
        platform = platform[1:]
        score = [i.text for i in bsobj.find_all('div', {'class': re.compile('metascore_w large game.*')})]
        score = score[::2] #since it extracts two scores at the same time
        #genre_type = re.findall(r'\w+', str(genre_name))
        #genre_type = str(genre_type).replace("['", "")
        #genre_type = str(genre_type).replace("']", "")
        genre = [genre_type for i in range(len(score))]
        titles += title
        platforms += platform
        scores += score
        genres += genre
        data = {'platform': platforms, 'genre': genres, 'title': titles, 'score': scores} 
    total_df = total_df.append(pd.DataFrame(data))
    total_df['platform'] = total_df['platform'].str.extract(r"( .*)")
    total_df['platform'] = total_df['platform'].str.replace('\n', '')
    #total_df['genre'] = total_df['genre'].str.extract(r"(\w+)")
    total_df.to_csv('metacritic_review_' + str(genre_type) + '.csv', encoding = 'utf-8')

if __name__ == '__main__':
    genres = ['adventure', 'fighting', 'first-person', 'flight', 'party', 'platformer', 'puzzle', 
    'racing', 'real-time', 'role-playing', 'simulation', 'sports', 'strategy', 'third-person',
    'turn-based', 'wargame', 'wrestling']
    for genre_name in zip(genres):
        metacritic_review(genre_name)



#%%
