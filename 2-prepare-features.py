import pandas as pd
import numpy as np
import pickle
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train = pd.read_csv("data/X_train.csv").squeeze()
X_test  = pd.read_csv("data/X_test.csv").squeeze()

# TF-IDF for SVM and Naive Bayes
tfidf = TfidfVectorizer(max_features=20000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

scipy.sparse.save_npz("data/X_train_tfidf.npz", X_train_tfidf)
scipy.sparse.save_npz("data/X_test_tfidf.npz",  X_test_tfidf)
with open("data/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Tokenized + padded sequences for RNN, LSTM, GRU
VOCAB_SIZE = 20000
MAX_LEN    = 200

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="pre", truncating="pre")
X_test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN, padding="pre", truncating="pre")

np.save("data/X_train_pad.npy", X_train_pad)
np.save("data/X_test_pad.npy",  X_test_pad)
with open("data/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("TF-IDF shape:    ", X_train_tfidf.shape)
print("Padded seq shape:", X_train_pad.shape)
