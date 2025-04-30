# ingestion/twitter_producer.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from kafka import KafkaProducer
import tweepy
from config import twitter_keys

# Kafka Producer setup
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Twitter API setup
auth = tweepy.OAuth1UserHandler(
    twitter_keys.API_KEY,
    twitter_keys.API_SECRET_KEY,
    twitter_keys.ACCESS_TOKEN,
    twitter_keys.ACCESS_TOKEN_SECRET
)
api = tweepy.API(auth)

sent_posts = set()

print("[INFO] Twitter Producer started... Will send real tweets whenever possible...")

while True:
    try:
        query = "news OR technology OR sports OR entertainment OR world"
        tweets = api.search_tweets(q=query, count=5, tweet_mode='extended')

        sent_in_this_round = False

        for tweet in tweets:
            text = tweet.full_text.strip()

            if text in sent_posts:
                continue  # Skip duplicates

            sent_posts.add(text)

            # Send to Kafka
            producer.send('twitter_stream', value={'text': text})
            producer.flush()

            print("[New Tweet Sent]:", text)
            sent_in_this_round = True

        if not sent_in_this_round:
            print("[INFO] No new tweets to send this time.")

        time.sleep(10)  # Short sleep between fetches

    except tweepy.errors.Forbidden as e:
        print("[Twitter API Forbidden]: Waiting 5 minutes before retrying...")
        time.sleep(300)  # 5 minutes sleep if forbidden

    except Exception as e:
        print("[Twitter Producer Error]:", e)
        time.sleep(30)  # Sleep a bit and retry
