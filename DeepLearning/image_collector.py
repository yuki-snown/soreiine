from requests_oauthlib import OAuth1Session
import json
import os
import re
import urllib
import setting

twitter = OAuth1Session(setting.CK, setting.CS, setting.AT, setting.AS)
mkdir_name = "images"

def dir_check(word):
    if not os.path.isdir(mkdir_name):
        os.mkdir(mkdir_name)
    check_count = 0
    while True:
        if not os.path.isdir(mkdir_name + "/" + word + str(check_count)):
            os.mkdir(mkdir_name + "/" + word + str(check_count))
            dir_name = "/"+ word + str(check_count)
            return dir_name
        check_count += 1

def get_target_ward(ward):
    url = "https://api.twitter.com/1.1/search/tweets.json"
    params = {'q':ward,
              'count':100
          }
    req = twitter.get(url, params = params)
    timeline = json.loads(req.text)
    return timeline

def get_illustration(timeline, dir_name):
    global image
    global image_number
    image_number = 0
    check_image = []
    for tweet in timeline['statuses']:
        try:
            media_list = tweet['extended_entities']['media']
            for media in media_list:
                image = media['media_url']
                if image in check_image:
                    continue
                with open(mkdir_name + dir_name +"/image_"+str(image_number) +"_"+os.path.basename(image), 'wb') as f:
                    img = urllib.request.urlopen(image).read()
                    f.write(img)
                check_image.append(image)
                image_number += 1
                print('saving')
        except:
            pass

if __name__ == '__main__':
    dir_name = dir_check("dir")
    wordlist = ["動物", "癒し", "可愛い", "かわいい", "綺麗", "自然", "エロい", "エッチ"]
    for word in wordlist:
        timeline = get_target_ward(word)
        get_illustration(timeline, dir_name)