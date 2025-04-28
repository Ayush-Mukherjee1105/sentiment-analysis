# ingestion/twitter_producer.py

import tweepy
from kafka import KafkaProducer
import json
from config import twitter_keys

# Connect to Kafka
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],   # Kafka server running locally
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # Encode to JSON
)

# Connect to Twitter API
client = tweepy.Client(bearer_token=twitter_keys.BEARER_TOKEN)

# Define the topic
KAFKA_TOPIC = 'twitter_stream'

# Search for recent tweets
query = 'technology OR elections OR sports -is:retweet lang:en'

def fetch_and_send_tweets():
    tweets = client.search_recent_tweets(query=query, max_results=10)
    if tweets.data:
        for tweet in tweets.data:
            print(f"Tweet: {tweet.text}")
            message = {
                'text': tweet.text
            }
            producer.send(KAFKA_TOPIC, value=message)
            producer.flush()

if __name__ == "__main__":
    fetch_and_send_tweets()
