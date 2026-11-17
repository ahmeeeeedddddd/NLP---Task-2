import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

X_train = np.load("data/X_train_pad.npy")
y_train = pd.read_csv("data/y_train.csv").squeeze()

VOCAB_SIZE = 20000

model = Sequential([
    Embedding(VOCAB_SIZE, 64, mask_zero=True),
    SimpleRNN(64),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
              metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)

start = time.time()
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, callbacks=[early_stop])
train_time = round(time.time() - start, 2)

model.save("models/rnn.keras")

with open("data/rnn_history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

try:
    times = json.load(open("data/train_times.json"))
except FileNotFoundError:
    times = {}
times["rnn"] = train_time
json.dump(times, open("data/train_times.json", "w"))

print(f"Simple RNN trained in {train_time}s")
