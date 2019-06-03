#create dataset of ipl matches tweets

import tweepy
import csv
import pandas as pd

# Input your credentials here
consumer_key=""
consumer_secret=""
access_token=""
access_token_secret=""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

# Enter search terms, if required
search_terms = '#ipl2019 OR #VIVOIPL OR #kkr OR #csk OR #mi OR #rcb OR #kxip OR #dc OR #srh OR #rr OR #Twitter IPL'

# Open/Create a file to append data
csvFile = open('ipldataset.csv','a')

# Use csv Writer
csvWriter = csv.writer(csvFile)

# Verification is optional
for tweet in tweepy.Cursor(api.search,q=search_terms,since='2019-03-01',lang="en").items(10):
      if (tweet.user.verified==True):
            print ("Tweet Created at:",tweet.user.created_at, "Tweet Text:", tweet.text)
            print("user id",tweet.id)
            print ("Name: ",tweet.user.name)
            print ("Screen Name: ",tweet.user.screen_name.encode("utf-8"))
            print ("Location: ",tweet.user.location)
            print ("Verification: ",tweet.user.verified)
            print ("__")
            csvWriter.writerow([tweet.user.created_at,tweet.id,tweet.user.name.encode("utf-8"),tweet.user.screen_name.encode("utf-8"),tweet.user.location.encode("utf-8"),tweet.text.encode("utf-8")])
