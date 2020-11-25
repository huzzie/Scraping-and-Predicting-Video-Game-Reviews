import requests

ff = open('res1.csv','w',encoding='utf-8')
ss = requests.session()

def spider_(id,NAME,topCriticScore,genres):
    data = []
    for i in range(0,50,10):
        url = "https://api.opencritic.com/api/review/game/{}?skip={}".format(id,i)
        response = ss.get(url).json()
        for res in response:
            try:
                name = res['Authors'][0]['name']
            except:
                name = '未知'
            try:
                score = res['score']
                npScore = res['npScore']
                platform = res['Platforms'][0]['name']
            except:
                score = None
                npScore = 100
                platform = None
            if score:
                line = ('{},{},{},{},{},{},{}\n'.format(NAME,topCriticScore,name,score,npScore,platform,'|'.join(genres)))
                ff.write(line)
    return data

def get_genres(id):
    genres = []
    url = 'https://api.opencritic.com/api/game/{}'.format(id)
    response = ss.get(url).json()
    genres_jsons = response['Genres']
    for genres_json in genres_jsons:
        genres.append(genres_json['name'])
    return (genres)


def main():
    for i in range(0,300,20):
        url = 'https://api.opencritic.com/api/game?skip={}'.format(i)
        print(url)
        response = ss.get(url).json()
        print(len(response))
        for ress in response:
            name = ress['name']
            url = ress['url']
            id = ress['id']
            topCriticScore = ress['topCriticScore']
            genres = get_genres(id)
            spider_(id,name,topCriticScore,genres)

if __name__ == "__main__":
    main()
    # get_genres(4504)