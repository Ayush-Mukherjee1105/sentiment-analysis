# streaming/kafka_consumer.py

import json
import os
import csv
from kafka import KafkaConsumer
import sys

# Path fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.infer import predict_sentiment

# Kafka setup
consumer = KafkaConsumer(
    'twitter_stream',
    'reddit_stream',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Ensure results directory
os.makedirs('results', exist_ok=True)
csv_path = 'results/predictions_live.csv'

# Always write header if file doesn't exist
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['text', 'platform', 'label'])

print("[INFO] Kafka Consumer started ✅")

# Consume messages
while True:
    for message in consumer:
        try:
            msg = message.value
            text = msg.get('text', '').strip().replace('"', '')
            platform = 'reddit' if message.topic == 'reddit_stream' else 'twitter'
            if not text:
                continue

            label = predict_sentiment(text)

            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow([text, platform, label])

            print(f"[{platform.upper()}] {label.upper()} — {text[:50]}...")

        except Exception as e:
            print("[Kafka Consumer Error]:", e)
