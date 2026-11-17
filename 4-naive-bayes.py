import pickle
import time
import json
import scipy.sparse
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

X_train = scipy.sparse.load_npz("data/X_train_tfidf.npz")
y_train = pd.read_csv("data/y_train.csv").squeeze()

start = time.time()
model = MultinomialNB()
model.fit(X_train, y_train)
train_time = round(time.time() - start, 2)

with open("models/naive_bayes.pkl", "wb") as f:
    pickle.dump(model, f)

try:
    times = json.load(open("data/train_times.json"))
except FileNotFoundError:
    times = {}
times["naive_bayes"] = train_time
json.dump(times, open("data/train_times.json", "w"))

print(f"Naive Bayes trained in {train_time}s")
