# ingestion/reddit_producer.py

import praw
from kafka import KafkaProducer
import json
from config import reddit_keys

# Connect to Kafka
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Connect to Reddit API
reddit = praw.Reddit(
    client_id=reddit_keys.CLIENT_ID,
    client_secret=reddit_keys.CLIENT_SECRET,
    user_agent=reddit_keys.USER_AGENT
)

KAFKA_TOPIC = 'reddit_stream'

def fetch_and_send_posts():
    subreddit = reddit.subreddit('worldnews')
    for post in subreddit.hot(limit=10):
        print(f"Reddit Post: {post.title}")
        message = {
            'text': post.title
        }
        producer.send(KAFKA_TOPIC, value=message)
        producer.flush()

if __name__ == "__main__":
    fetch_and_send_posts()
