import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
import string
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preprocessing function matching training
stop_words = set(stopwords.words("english")) - {"no", "not", "nor"}
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in word_tokenize(text) if w not in stop_words]
    return " ".join(tokens)

# Load feature extractors and models
with open("data/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/svm.pkl", "rb") as f:
    svm = pickle.load(f)
with open("models/naive_bayes.pkl", "rb") as f:
    nb = pickle.load(f)

rnn  = tf.keras.models.load_model("models/rnn.keras")
lstm = tf.keras.models.load_model("models/lstm.keras")
gru  = tf.keras.models.load_model("models/gru.keras")

unseen_texts = [
    "This was one of the most brilliant and masterfully executed films of the decade.",
    "Completely unwatchable, dull script and atrocious acting from start to finish.",
    "Not bad at all, though it had some pacing issues in the middle section.",
    "I expected nothing and was still disappointed by how boring this movie turned out to be.",
    "A heartwarming story with stunning visuals and an unforgettable soundtrack."
]

label_map = {1: "Positive", 0: "Negative"}
rows = []

for raw in unseen_texts:
    cleaned = clean_text(raw)
    feat_tfidf = tfidf.transform([cleaned])
    feat_pad = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=200, padding="pre", truncating="pre")
    
    p_svm = int(svm.predict(feat_tfidf)[0])
    p_nb  = int(nb.predict(feat_tfidf)[0])
    p_rnn = int((rnn.predict(feat_pad, verbose=0)[0][0] > 0.5))
    p_lstm = int((lstm.predict(feat_pad, verbose=0)[0][0] > 0.5))
    p_gru  = int((gru.predict(feat_pad, verbose=0)[0][0] > 0.5))
    
    rows.append({
        "Sample Review": raw,
        "SVM": label_map[p_svm],
        "Naive Bayes": label_map[p_nb],
        "Simple RNN": label_map[p_rnn],
        "LSTM": label_map[p_lstm],
        "GRU": label_map[p_gru]
    })

df_res = pd.DataFrame(rows)
print("\n" + "="*80)
print("PREDICTIONS ON UNSEEN TEXTS")
print("="*80)
for idx, r in df_res.iterrows():
    print(f"\nReview #{idx+1}: {r['Sample Review']}")
    print(f"  Predictions -> SVM: {r['SVM']} | NB: {r['Naive Bayes']} | RNN: {r['Simple RNN']} | LSTM: {r['LSTM']} | GRU: {r['GRU']}")

df_res.to_csv("data/unseen_predictions.csv", index=False)
