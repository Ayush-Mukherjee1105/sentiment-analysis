# ingestion/reddit_producer.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from kafka import KafkaProducer
import praw
from config import reddit_keys

# Kafka Producer setup
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Reddit API setup (READ ONLY)
reddit = praw.Reddit(
    client_id=reddit_keys.CLIENT_ID,
    client_secret=reddit_keys.CLIENT_SECRET,
    user_agent=reddit_keys.USER_AGENT
)

sent_posts = set()

print("[INFO] Reddit Producer started... Fetching real Reddit posts...")

while True:
    try:
        # Pull hot posts from a popular subreddit like 'funny'
        subreddit = reddit.subreddit('all')
        posts = subreddit.new(limit=5)

        for submission in posts:
            text = submission.title.strip()

            if text in sent_posts:
                print("[Duplicate] Skipping already sent Reddit post.")
                continue

            sent_posts.add(text)

            # Send to Kafka
            producer.send('reddit_stream', value={'text': text})
            producer.flush()

            print("[New Reddit Post Sent]:", text)

        time.sleep(10)

    except Exception as e:
        print("[Reddit Producer Error]:", e)
        time.sleep(10)
