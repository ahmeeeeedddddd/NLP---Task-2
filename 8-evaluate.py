import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

X_test_tfidf = scipy.sparse.load_npz("data/X_test_tfidf.npz")
X_test_pad   = np.load("data/X_test_pad.npy")
y_test       = pd.read_csv("data/y_test.csv").squeeze()

with open("models/svm.pkl", "rb") as f:
    svm = pickle.load(f)
with open("models/naive_bayes.pkl", "rb") as f:
    nb = pickle.load(f)

rnn  = tf.keras.models.load_model("models/rnn.keras")
lstm = tf.keras.models.load_model("models/lstm.keras")
gru  = tf.keras.models.load_model("models/gru.keras")

preds = {
    "SVM": svm.predict(X_test_tfidf),
    "Naive Bayes": nb.predict(X_test_tfidf),
    "Simple RNN": (rnn.predict(X_test_pad, verbose=0) > 0.5).astype(int).flatten(),
    "LSTM": (lstm.predict(X_test_pad, verbose=0) > 0.5).astype(int).flatten(),
    "GRU": (gru.predict(X_test_pad, verbose=0) > 0.5).astype(int).flatten()
}

results = {}
target_names = ["Negative (0)", "Positive (1)"]

for name, y_pred in preds.items():
    acc = accuracy_score(y_test, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    
    results[name] = {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4)
    }
    
    print(f"\n{'='*20} {name} {'='*20}")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

with open("data/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Plot DL train/val curves
for model_name in ["rnn", "lstm", "gru"]:
    path = f"data/{model_name}_history.json"
    if os.path.exists(path):
        hist = json.load(open(path))
        acc_key = "accuracy" if "accuracy" in hist else "train_accuracy"
        val_acc_key = "val_accuracy" if "val_accuracy" in hist else None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        if acc_key in hist:
            ax1.plot(hist[acc_key], label="Train Acc")
        if val_acc_key and val_acc_key in hist:
            ax1.plot(hist[val_acc_key], label="Val Acc")
        ax1.set_title(f"{model_name.upper()} Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        
        if "loss" in hist:
            ax2.plot(hist["loss"], label="Train Loss")
        if "val_loss" in hist:
            ax2.plot(hist["val_loss"], label="Val Loss")
        ax2.set_title(f"{model_name.upper()} Loss")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"plots/{model_name}_learning_curves.png")
        plt.close()

print("\nEvaluation completed and saved to data/evaluation_results.json and plots/")
