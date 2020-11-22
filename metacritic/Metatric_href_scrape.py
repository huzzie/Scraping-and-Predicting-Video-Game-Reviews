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

# %%
#driver=webdriver.Chrome(executable_path=r"/Users/HyunjaeCho/Documents/Fall2020/DataMining/chromedriver")
#games_site = 'https://www.metacritic.com/game/playstation-4'

#%%
game_lists = ['playstation-4', 'playstation-5', 'xbox-one', 'xbox-series-x', 'switch', 'pc']



#%%
def game_hrefs(game_type):
    all_hrefs = []
    for i in game_lists:
        driver=webdriver.Chrome(executable_path=r"/Users/HyunjaeCho/Documents/Fall2020/DataMining/chromedriver")
        game_site = 'https://www.metacritic.com/game/' + str(i)
        driver.get(game_site)
        driver.implicitly_wait(5)
        html = driver.page_source
        bsobj = BeautifulSoup(html, 'html.parser')
        meta_list = bsobj.find('table', {'class' : 'clamp-list'}).findAll('td', {'class': 'clamp-image-wrap'})
        ### Metatric href
        meta_href = []
        for i in meta_list:
            href = i.find('a').get('href')
            meta_href.append(href)
        # before we move on to user href, let's take a moment 
        driver.implicitly_wait(10)
        driver.find_element_by_xpath("//a[@class='partial']").click()
        driver.implicitly_wait(1)

        ### User href
        user_list = bsobj.find('table', {'class' : 'clamp-list'}).findAll('td', {'class': 'clamp-image-wrap'})
        user_href = []
        for j in user_list:
            href2 = j.find('a').get('href')
            user_href.append(href2)
        data = {'meta_href' : meta_href, 'user_href': user_href}
        data = pd.DataFrame(data)
        all_hrefs.append(data)
        #print(data)
    all_hrefs = pd.concat(all_hrefs)
    all_hrefs.to_csv('metacritic.href.csv', encoding = 'utf-8')
#%%
game_hrefs(game_lists)
# %%