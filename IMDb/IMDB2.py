# -*- coding: utf-8 -*-
import re
import json
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

def get_html(url):
    response=requests.get(url)
    if response.status_code==200:
        return  response.text
    else:
        return None

def parse_html(html):
    soup = BeautifulSoup(html, 'lxml')
    movies = soup.select('tbody tr')
    for movie in movies:
        poster = movie.select_one('.posterColumn')
        score = poster.select_one('span[name="ir"]')['data-value']
        movie_link = movie.select_one('.titleColumn').select_one('a')['href']
        year_str = movie.select_one('.titleColumn').select_one('span').get_text()
        year_pattern = re.compile('\d{4}')
        year = int(year_pattern.search(year_str).group())
        id_pattern = re.compile(r'(?<=tt)\d+(?=/?)')
        movie_id = int(id_pattern.search(movie_link).group())
        movie_name = movie.select_one('.titleColumn').select_one('a').string
        yield {
            'movie_id': movie_id,
            'movie_name': movie_name,
            'year': year,
            'movie_link': movie_link,
            'movie_rate': float(score)
        }

def write_file(content):
    with open('movie12.txt','a',encoding='utf-8')as f:
        f.write(json.dumps(content,ensure_ascii=False)+'\n')

def main():
    url='https://www.imdb.com/chart/top'
    html=get_html(url)
    for item in parse_html(html):
        write_file(item)

if __name__ == '__main__':
    main()