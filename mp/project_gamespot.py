#%%
from logging import raiseExceptions
import pandas
from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from time import sleep
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

#%%
gamespot_scores = pd.DataFrame()

#%%
driver = webdriver.Chrome(r'/Users/perry/Desktop/github_mp/dev/chromedriver_v85')

driver.get('https://www.gamespot.com/games/reviews/')

platform_list = []

#%%
#Set up the page to filter to what we want
select_platform = driver.find_element_by_xpath('//*[@id="review_filter_type_platform"]').click()
select_platform
sleep(0.1)

select_ps4 = driver.find_element_by_xpath('//*[@id="review_filter_type_platform"]/option[7]').click()
select_ps4
sleep(0.1)

select_genre = driver.find_element_by_xpath('//*[@id="review_filter_type_genre"]').click()
select_genre
sleep(0.1)

select_action = driver.find_element_by_xpath('//*[@id="review_filter_type_genre"]/option[5]').click()
select_action
sleep(0.1)

refresh = driver.find_element_by_xpath('//*[@id="reviews_filter_form"]/ul/li[5]/input').click()
sleep(0.1)

order_button = driver.find_element_by_xpath('//*[@id="review_door_sort"]').click()
order_button
sleep(0.1)

select_order = driver.find_element_by_xpath('//*[@id="review_door_sort"]/option[2]').click()
select_order
sleep(0.1)

#%%
# Get the scores from the page
game_class = 'horizontal-card-item'
reviews = {}
sleep(1)

#%%
number_of_pages = driver.find_element_by_xpath('//*[@id="js-sort-filter-results"]/ul/li[8]/a').text
number_of_pages = int(number_of_pages)
count_games = 0 
missed = 0
# print(number_of_pages)
# driver.close()

#%%

# Go through each page and get the title and score

for i in range(number_of_pages - 1):
    for game in driver.find_elements_by_class_name(game_class):
        try:
            game_title = game.find_element_by_class_name('horizontal-card-item__title  ').text
            rating = game.find_element_by_class_name('review-ring-score__score').text
            reviews[game_title] = rating
            count_games += 1
        except:
            missed += 1
    print('Scraped page: ', i+1)
    sleep(1)
    number_of_pages = driver.find_element_by_class_name('next').click()
    sleep(1)

print(reviews)
print('Could not get', missed, 'of %i' % (missed + count_games))

#%%

genre_list = ['Action', 'Adventure', 'Fighting', 'First-Person', 'Flight', 'Party/Minigame', 'Platformer', 'Puzzle', 'Driving/Racing', 'Real-Time', 'Role-Playing', 'Simulation', 'Sports', 'Strategy', 'Third-Person', 'Turn-Based', 'Wrestling']
platform_list = ['3DS', 'Nintendo Switch', 'PC', 'PlayStation 3', 'PlayStation 4', 'PlayStation Vita', 'Wii', 'Wii U', 'Xbox 360', 'Xbox One']

#%%
driver.close()

#%%
# Initialize the driver and open the webpage
driver = webdriver.Chrome(r'/Users/perry/Desktop/github_mp/dev/chromedriver_v85')
driver.maximize_window()
driver.get('https://www.gamespot.com/games/reviews/')

#%%

def getReviews(platform, genre):
    # Throw exceptions
    
    # This will allow for the function to be expanded to all the valid values on Gamespot's website. For now, just using the platforms and genres we're interested in
    platform_list = ['3DS', 'Nintendo Switch', 'PC', 'PlayStation 3', 'PlayStation 4', 'PlayStation Vita', 'Wii', 'Wii U', 'Xbox 360', 'Xbox One']
    if platform not in platform_list:
        raise Exception('Platform input is not a selectable platform on Gamespot')
    
    genre_list = ['Action', 'Adventure', 'Fighting', 'First-Person', 'Flight', 'Party/Minigame', 'Platformer', 'Puzzle', 'Driving/Racing', 'Real-Time', 'Role-Playing', 'Simulation', 'Sports', 'Strategy', 'Third-Person', 'Turn-Based', 'Wrestling']
    if genre not in genre_list:
        raise Exception('Genre input is not a selectable genre on Gamespot')
    
    # Set up the page to filter to what we want
    
    # Open the platform dropdown
    open_platform_dropdown = driver.find_element_by_xpath('//*[@id="review_filter_type_platform"]').click()
    open_platform_dropdown
    sleep(0.1)
    
    # Select the platform parameter
    select_platform = Select(driver.find_element_by_id('review_filter_type_platform'))
    select_platform.select_by_visible_text(platform)

    # Open the genre dropdown
    select_genre_dropdown = driver.find_element_by_xpath('//*[@id="review_filter_type_genre"]').click()
    select_genre_dropdown
    sleep(0.1)
    
    # Select the genre
    select_platform = Select(driver.find_element_by_id('review_filter_type_genre'))
    select_platform.select_by_visible_text(genre)
    
    # Refresh the reviews to get the selected options
    refresh = driver.find_element_by_xpath('//*[@id="reviews_filter_form"]/ul/li[5]/input').click()
    refresh 
    sleep(0.1)
    
    # Order reviews from highest to lowest (not essential)
    order_button = driver.find_element_by_xpath('//*[@id="review_door_sort"]').click()
    order_button
    
    sleep(0.1)

    order_highest_to_lowest = driver.find_element_by_xpath('//*[@id="review_door_sort"]/option[2]').click()
    order_highest_to_lowest
    sleep(1)
    
    # Get the number of pages to loop through, if applicable
    try:
        number_of_pages = driver.find_element_by_xpath('//*[@id="js-sort-filter-results"]/ul/li[last()-1]').text
        number_of_pages = int(number_of_pages)
        # print(number_of_pages)
    except:
        number_of_pages = 2
    
    count_games = 0 
    missed = 0
    
    # Class of the div that will be looped through for all reviews
    game_class = 'horizontal-card-item'
    reviews = {}
    sleep(1)
    
    print('Beginning scrapping', genre, 'reviews for', platform, '...\n')
    for i in range(number_of_pages - 1):
        for game in driver.find_elements_by_class_name(game_class):
            try:
                game_title = game.find_element_by_class_name('horizontal-card-item__title  ').text
                rating = game.find_element_by_class_name('review-ring-score__score').text
                reviews[game_title] = rating
                count_games += 1
            except:
                missed += 1
        print('Scraped page: ', i+1)
        sleep(1)
        try:
            next = driver.find_element_by_class_name('next').click()
            next
        except:
            pass
        sleep(1)

    # print(reviews)
    print('Retrieved', count_games, 'of %i reviews\n' % (missed + count_games))
    
    # Return to homepage
    # select_reviews_nav = driver.find_element_by_xpath('//*[@id="masthead"]/div[2]/div[3]/div/div[2]/nav[1]/ul[2]/li[2]/div[1]/a').click()
    # select_reviews_nav
    # sleep(0.1)
    
    # go_home = driver.find_element_by_xpath('//*[@id="masthead"]/div[2]/div[3]/div/div[2]/nav[1]/ul[2]/li[2]/div[2]/div/ul/li[1]/a').click()
    # go_home
    # sleep(1)
    
    return reviews

#%%
# columns = platform_list
# data = np.array([np.arange(10)]*len(columns)).T
gamespot_reviews = pd.DataFrame(columns=['platform', 'genre', 'title', 'score'])
# gamespot_reviews = gamespot_reviews.fillna(0)
print(gamespot_reviews.head())

#%%

for platform in platform_list[0:2]:
    # for genre in genre_list[0:2]:
    output = getReviews(platform, 'Action')
    for key, value in output.items():
        gamespot_reviews = gamespot_reviews.append({'platform': platform, 'genre': 'Action', 'title': key, 'score': value}, ignore_index=True)

print(gamespot_reviews.head())

#%%
def getAllReviews(platformList, genreList):
    
    output = {}
    
    for platform in platformList:
        for genre in genreList:
            output = getReviews(platform, genre)
            
    return output
#%%
all_reviews = getAllReviews(platform_list, genre_list)

#%%

def buildData():
    gamespot_reviews = pd.DataFrame(columns=['platform', 'genre', 'title', 'score'])
    
    all_reviews = getAllReviews(platform_list, genre_list)
    for key, value in all_reviews.items():
        gamespot_reviews = gamespot_reviews.append({'platform': platform, 'genre': genre, 'title': key, 'score': value}, ignore_index=True)
    return gamespot_reviews

#%%
buildData()

#%%

# 11/23
# Ran overnight and PC, Action scraped all 1554 reviews from 74 pages, but got the error :
# ElementClickInterceptedException: Message: element click intercepted: Element <select id="review_filter_type_platform" name="review_filter_type[platform]">...</select> is not clickable at point (1195, 11). Other element would receive the click: <nav class="js-masthead-site-nav masthead-nav-section masthead-site-nav flex-grow">...</nav>

#%%

print(gamespot_reviews.head())

#%%

#%%
# Write a function to use the above function to go through each genre and save each dictionary to a dictionary for the genre. Output a dictionary of dictionaries

# gamespot_reviews = {}
platform = {}
genre = {}

for platform in platform_list:
    if platform not in platform.keys():
        for genre in genre_list:
            genre_list[genre] = getReviews(platform, genre)
        platform[platform] = genre

#%% [markdown]

# Output data
# {platform: 
#           {genre:
#                   {game: review score}
#           {
# }

#%% [markdown]
# list of metacritic genres

# action
# adventure
# fighting games
# first-person shooters
# flight/flying
# party
# platformer
# puzzle
# racing
# real-time strategy
# role-playing
# simulation
# sports
# strategy
# third-person shooter
# turn-based strategy
# wargames
# wrestling

#%%
# Gamespot genres

# Action
# Adventure
# Fighting
# First-Person
# Flight
# Party/Minigame
# Platformer
# Puzzle
# Driving/Racing
# Real-Time
# Role-Playing
# Simulation
# Sports (also has Baseball, Basketball, Billiards, Boxing, Football (American), etc.)
# Strategy
# Third-Person
# Turn-Based
# N/A for wargames (games listed in metacritic's wargames genre as spread between other genres on Gamespot)
# Wrestling