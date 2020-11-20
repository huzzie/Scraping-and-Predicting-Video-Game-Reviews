#%%
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

#%%
metacritic_scores = pd.DataFrame()

#%%
driver = webdriver.Chrome(r'/Users/perry/Desktop/github_mp/dev/chromedriver_v85')

driver.get('https://www.metacritic.com/game')

# %%
select_ps4 = driver.find_element_by_xpath('//*[@id="main"]/div[1]/div[2]/div/div[2]/div/ul/li[1]/div/span/a')
sleep(0.1)
select_ps4.click()
sleep(0.1)
select_ps4_action = driver.find_element_by_xpath('//*[@id="main"]/div[4]/div/div[2]/div[2]/div[1]/div/div/div/ul/li[1]/a')
sleep(0.1)
select_ps4_action.click()
sleep(0.1)
driver.quit()

# div can either be
# 1. metascore_w large game positive
# 2. metascore_w large game mixed
# 3. metascore_w large game negative


# %%
#main > div.module.new_releases_module > div.body > div > div.platforms.current_platforms > div > ul > li.platform.first_platform > div > span > a