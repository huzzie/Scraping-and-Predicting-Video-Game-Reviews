#%%
from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from time import sleep
import pandas as pd
import numpy as np

#%%
genre_list = ['Action', 'Adventure', 'Fighting', 'First-Person', 'Flight', 'Party/Minigame', 'Platformer', 'Puzzle', 'Driving/Racing', 'Real-Time', 'Role-Playing', 'Simulation', 'Sports', 'Strategy', 'Third-Person', 'Turn-Based', 'Wrestling']
platform_list = ['3DS', 'Nintendo Switch', 'PC', 'PlayStation 3', 'PlayStation 4', 'PlayStation Vita', 'Wii', 'Wii U', 'Xbox 360', 'Xbox One']

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
    sleep(0.5)
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
    sleep(0.5)
    
    # Get the number of pages to loop through, if applicable
    try:
        number_of_pages = driver.find_element_by_xpath('//*[@id="js-sort-filter-results"]/ul/li[last()-1]').text
        number_of_pages = int(number_of_pages)
        # print(number_of_pages)
    except:
        number_of_pages = 2
    
    # Class of the div that will be looped through for all reviews
    game_class = 'horizontal-card-item'
    reviews = {}
    count_games = 0 
    missed = 0
    
    print('Beginning scraping', genre, 'reviews for', platform, '...\n')
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
        try:
            next = driver.find_element_by_class_name('next').click()
            next
        except:
            pass
        sleep(0.5)

    # print(reviews)
    print('Retrieved', count_games, 'of %i reviews\n' % (missed + count_games))
    
    return reviews

#%%
def getAllReviews(platformList, genreList):
    
    gamespot_reviews = pd.DataFrame(columns=['platform', 'genre', 'title', 'score'])
    
    for platform in platformList:
        for genre in genreList:
            all_reviews = getReviews(platform, genre)
            for key, value in all_reviews.items():
                gamespot_reviews = gamespot_reviews.append({'platform': platform, 'genre': genre, 'title': key, 'score': value}, ignore_index=True)
            
    return gamespot_reviews

#%%
gamespot_reviews_df = getAllReviews(platform_list, genre_list)

#%%
print(gamespot_reviews_df)

#%%
gamespot_reviews_df.to_csv('gamespot_reviews.csv', index=False)

#%%
# Would like to write another function to append more data to the dataframe/csv 

#%%
driver.close()