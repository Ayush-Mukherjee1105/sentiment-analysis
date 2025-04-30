# 📜 Steps to Run the Project 

---

# 🚀 Real-Time Sentiment Analysis on Social Media
**Using Distributed NLP and Context-Aware Deep Learning**

---

## 📦 Project Setup Instructions

---

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd sentiment-analysis
```

---

### 2. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

_(No version pinning — latest stable packages will be installed.)_

---

### 3. Setup Twitter and Reddit API Access

- Create a **Twitter Developer account** and generate Bearer Token.
- Create a **Reddit script app** and get Client ID, Client Secret, and User Agent.

Store API credentials inside:

- `config/twitter_keys.py`
- `config/reddit_keys.py`

Example format:

```python
# twitter_keys.py
API_KEY = 'your-api-key'
API_SECRET_KEY = 'your-api-secret'
BEARER_TOKEN = 'your-bearer-token'
ACCESS_TOKEN = 'your-access-token'
ACCESS_TOKEN_SECRET = 'your-access-token-secret'
```

```python
# reddit_keys.py
CLIENT_ID = 'your-client-id'
CLIENT_SECRET = 'your-client-secret'
USER_AGENT = 'your-user-agent'
```

---

### 4. Start Apache Kafka Server

Kafka is used for real-time message streaming.

1. Start **Zookeeper Server**:

```bash
cd kafka_2.13-3.5.0
bin\windows\zookeeper-server-start.bat config\zookeeper.properties
```

2. Start **Kafka Broker** manually (because WMIC is deprecated):

```bash
java -cp "libs\*" -Xmx1G -Xms1G kafka.Kafka config\server.properties
```

---

### 5. Run the Producers (Twitter and Reddit)

Open separate terminals for each producer.

✅ Start Twitter Producer:

```bash
python ingestion/twitter_producer.py
```

✅ Start Reddit Producer:

```bash
python ingestion/reddit_producer.py
```

Both producers will send live tweets and reddit posts into Kafka topics (`twitter_stream` and `reddit_stream`).

---

### 6. Preprocessing

Incoming social media posts are cleaned using:

- `preprocessing/clean_text.py` → Removes URLs, mentions, hashtags, emojis, stopwords, etc.
- `preprocessing/eda_utils.py` → Collects live EDA (word frequency, post length analysis).

---

### 7. Sentiment Analysis Model Setup

Sentiment prediction is performed using:

- **Lightweight DistilBERT + Quantization**
- Optimized for **GPU acceleration** (if available)
- Real-time inference using `model/infer.py`
- Outputs **positive**, **negative**, or **neutral** labels.

---

### 8. Kafka Consumer (Real-Time Pipeline)

The consumer:

- Reads live posts from Kafka topics
- Cleans the text
- Predicts sentiment
- Prints live results to terminal

✅ Consumer will be run using:

```bash
python streaming/kafka_consumer.py
```

---

# ✅ Current Status

| Component | Status |
|:---|:---|
| Kafka Broker | ✅ Running |
| Twitter Producer | ✅ Working |
| Reddit Producer | ✅ Working |
| Preprocessing | ✅ Implemented |
| EDA Tools | ✅ Implemented |
| Sentiment Model | ✅ Built (DistilBERT quantized version coming next) |
| Real-Time Consumer | 🔜 Building now |

---

# ✨ Technologies Used

- **Apache Kafka** — Real-time streaming
- **Tweepy, PRAW** — API connections
- **PyTorch** — Deep learning (DistilBERT + BiLSTM)
- **Transformers (Huggingface)** — NLP models
- **Optimum, ONNXRuntime** — Model optimization
- **Dash (Future)** — Visualization dashboard
- **Flask (Future)** — Backend server

---

# 📅 Next Steps (After Current)

- Finish optimized `infer.py`
- Update Kafka Consumer to call real model
- Add Real-Time Dashboard using Dash (Optional)

