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
select_ps4 = driver.find_element_by_xpath('//*[@id="main"]/div[1]/div[2]/div/div[2]/div/ul/li[1]/div/span/a').click()
select_ps4
sleep(0.1)
select_ps4_action = driver.find_element_by_xpath('//*[@id="main"]/div[4]/div/div[2]/div[2]/div[1]/div/div/div/ul/li[1]/a').click()
select_ps4_action
sleep(0.1)
#%%
first_score = driver.find_element_by_xpath('//*[@id="main_content"]/div[1]/div[2]/div/div[1]/div/div[2]/table/tbody/tr[1]/td[2]/div[1]/a/div').text

second_score = driver.find_element_by_xpath('//*[@id="main_content"]/div[1]/div[2]/div/div[1]/div/div[2]/table/tbody/tr[3]/td[2]/div[1]/a/div').text

third_score = driver.find_element_by_xpath('//*[@id="main_content"]/div[1]/div[2]/div/div[1]/div/div[2]/table/tbody/tr[5]/td[2]/div[1]/a/div')

fourth_score = driver.find_element_by_xpath('//*[@id="main_content"]/div[1]/div[2]/div/div[1]/div/div[2]/table/tbody/tr[7]/td[2]/div[1]/a/div')

fifth_score = driver.find_element_by_xpath('//*[@id="main_content"]/div[1]/div[2]/div/div[1]/div/div[2]/table/tbody/tr[9]/td[2]/div[1]/a/div').text

sixth_score = driver.find_element_by_xpath('//*[@id="main_content"]/div[1]/div[2]/div/div[1]/div/div[4]/table/tbody/tr[1]/td[2]/div[1]/a/div')

tenth_score = driver.find_element_by_xpath('//*[@id="main_content"]/div[1]/div[2]/div/div[1]/div/div[4]/table/tbody/tr[9]/td[2]/div[1]/a/div').text

eleventh_score = driver.find_element_by_xpath('//*[@id="main_content"]/div[1]/div[2]/div/div[1]/div/div[6]/table/tbody/tr[1]/td[2]/div[1]/a/div')

second_to_last_score = driver.find_element_by_xpath('//*[@id="main_content"]/div[1]/div[2]/div/div[1]/div/div[8]/table/tbody/tr[167]/td[2]/div[1]/a/div')

last_score = driver.find_element_by_xpath('//*[@id="main_content"]/div[1]/div[2]/div/div[1]/div/div[8]/table/tbody/tr[169]/td[2]/div[1]/a/div').text

#%%
score_xpath_prefix = '//*[@id="main_content"]/div[1]/div[2]/div/div[1]/div/div['
div = ''
score_xpath_middle = ']/table/tbody/tr['
tr = ''
score_xpath_suffix = ']/td[2]/div[1]/a/div'

score_list = []

for div_number in range(2, 8, 10):
    div = str(div_number)
    print(div)
    for tr_number in range(1, 11, 2):
        tr = str(tr_number)
        print(tr)
        try:
            score_list.append(driver.find_element_by_xpath(score_xpath_prefix + div + score_xpath_middle + tr + score_xpath_suffix).text)
        except:
            print("Couldn't find score.")

print(score_list)

#%%
driver.close()
print('Scraping complete.')

#%%
test = driver.find_element_by_class_name('metascore_w large game positive').text

test2 = driver.find_elements_by_css_selector('#main_content > div.browse.new_releases > div.content_after_header > div > div.next_to_side_col > div > div.browse_list_wrapper.one.browse-list-large > table > tbody > tr:nth-child(9) > td.clamp-summary-wrap > div.clamp-score-wrap > a > div')

#main_content > div.browse.new_releases > div.content_after_header > div > div.next_to_side_col > div > div.browse_list_wrapper.one.browse-list-large > table > tbody > tr:nth-child(9) > td.clamp-summary-wrap > div.clamp-score-wrap > a > div')

# div can either be
# 1. metascore_w large game positive
# 2. metascore_w large game mixed
# 3. metascore_w large game negative


# %%
#main > div.module.new_releases_module > div.body > div > div.platforms.current_platforms > div > ul > li.platform.first_platform > div > span > a