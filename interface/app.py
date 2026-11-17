import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure we can find the parent folder's models, data, and plots
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)

import re
import string
import json
import pickle
import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preprocessing
stop_words = set(stopwords.words("english")) - {"no", "not", "nor"}
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in word_tokenize(text) if w not in stop_words]
    return " ".join(tokens)

# Load artifacts
tfidf = pickle.load(open("data/tfidf.pkl", "rb")) if os.path.exists("data/tfidf.pkl") else None
tokenizer = pickle.load(open("data/tokenizer.pkl", "rb")) if os.path.exists("data/tokenizer.pkl") else None

svm = pickle.load(open("models/svm.pkl", "rb")) if os.path.exists("models/svm.pkl") else None
nb = pickle.load(open("models/naive_bayes.pkl", "rb")) if os.path.exists("models/naive_bayes.pkl") else None

rnn = tf.keras.models.load_model("models/rnn.keras") if os.path.exists("models/rnn.keras") else None
lstm = tf.keras.models.load_model("models/lstm.keras") if os.path.exists("models/lstm.keras") else None
gru = tf.keras.models.load_model("models/gru.keras") if os.path.exists("models/gru.keras") else None

def predict_single(text, chosen_model):
    if not text.strip():
        return "Please enter a valid review text."
    
    cleaned = clean_text(text)
    
    if chosen_model in ["SVM", "Naive Bayes"]:
        if tfidf is None:
            return "TF-IDF vectorizer not loaded."
        feat = tfidf.transform([cleaned])
        if chosen_model == "SVM":
            if svm is None: return "SVM model not trained yet."
            pred = int(svm.predict(feat)[0])
            score = float(svm.decision_function(feat)[0])
            label = "Positive" if pred == 1 else "Negative"
            return f"Model: {chosen_model}\nPredicted Sentiment: {label}\nDecision Score: {score:+.4f}"
        else:
            if nb is None: return "Naive Bayes model not trained yet."
            pred = int(nb.predict(feat)[0])
            proba = nb.predict_proba(feat)[0]
            label = "Positive" if pred == 1 else "Negative"
            return f"Model: {chosen_model}\nPredicted Sentiment: {label}\nConfidence: {proba[pred]*100:.2f}% (Pos: {proba[1]*100:.2f}%, Neg: {proba[0]*100:.2f}%)"
    
    else:
        if tokenizer is None:
            return "Tokenizer not loaded."
        seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=200, padding="pre", truncating="pre")
        
        m_map = {"Simple RNN": rnn, "LSTM": lstm, "GRU": gru}
        dl_model = m_map.get(chosen_model)
        if dl_model is None:
            return f"{chosen_model} model not trained yet."
        
        p = float(dl_model.predict(seq, verbose=0)[0][0])
        label = "Positive" if p > 0.5 else "Negative"
        conf = p if p > 0.5 else (1.0 - p)
        return f"Model: {chosen_model}\nPredicted Sentiment: {label}\nProbability Score: {p:.4f} (Confidence: {conf*100:.2f}%)"

def predict_all(text):
    if not text.strip():
        return pd.DataFrame()
    
    cleaned = clean_text(text)
    rows = []
    
    # Linear models
    if tfidf is not None:
        feat = tfidf.transform([cleaned])
        if svm is not None:
            p = int(svm.predict(feat)[0])
            rows.append({"Model": "SVM (LinearSVC)", "Predicted Sentiment": "Positive" if p == 1 else "Negative", "Score / Probability": f"{svm.decision_function(feat)[0]:+.4f}"})
        if nb is not None:
            p = int(nb.predict(feat)[0])
            proba = nb.predict_proba(feat)[0]
            rows.append({"Model": "Naive Bayes (MultinomialNB)", "Predicted Sentiment": "Positive" if p == 1 else "Negative", "Score / Probability": f"{proba[1]:.4f}"})
            
    # DL models
    if tokenizer is not None:
        seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=200, padding="pre", truncating="pre")
        for name, m in [("Simple RNN", rnn), ("LSTM", lstm), ("GRU", gru)]:
            if m is not None:
                p = float(m.predict(seq, verbose=0)[0][0])
                rows.append({"Model": name, "Predicted Sentiment": "Positive" if p > 0.5 else "Negative", "Score / Probability": f"{p:.4f}"})
                
    return pd.DataFrame(rows)

def get_comparison_table():
    if os.path.exists("data/model_comparison.csv"):
        return pd.read_csv("data/model_comparison.csv")
    return pd.DataFrame([{"Status": "Run 8-evaluate.py and 9-compare.py to populate metrics."}])

def get_classification_reports():
    if os.path.exists("data/evaluation_results.json"):
        res = json.load(open("data/evaluation_results.json"))
        return json.dumps(res, indent=4)
    return "Evaluation results not generated yet. Please run 8-evaluate.py."

# Gradio Interface
with gr.Blocks(title="NLP Task 2: Multi-Model Evaluation & Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 NLP Task 2: Model Comparison & Unseen Text Evaluation Dashboard")
    gr.Markdown("Fairly evaluating and comparing **SVM, Naive Bayes, Simple RNN, LSTM, and GRU** on the shared dataset.")
    
    with gr.Tab("🎯 Predict with Specific Model (Unseen Text)"):
        with gr.Row():
            with gr.Column():
                input_single = gr.Textbox(
                    label="Enter Text Review / Statement",
                    lines=4,
                    value="This film was an absolute masterpiece with astonishing performances and brilliant direction."
                )
                model_dropdown = gr.Dropdown(
                    choices=["SVM", "Naive Bayes", "Simple RNN", "LSTM", "GRU"],
                    value="SVM",
                    label="Select Model to Test"
                )
                btn_single = gr.Button("Predict with Selected Model", variant="primary")
            with gr.Column():
                output_single = gr.Textbox(label="Single Model Prediction Result", lines=6)
        btn_single.click(fn=predict_single, inputs=[input_single, model_dropdown], outputs=output_single)
        
    with gr.Tab("⚡ Multi-Model Comparative Prediction"):
        input_all = gr.Textbox(
            label="Enter Text Review / Statement",
            lines=3,
            value="The pacing in the second act was sluggish, but the ending completely redeemed it."
        )
        btn_all = gr.Button("Analyze Across All 5 Models", variant="secondary")
        output_all = gr.Dataframe(label="Side-by-Side Model Predictions")
        btn_all.click(fn=predict_all, inputs=input_all, outputs=output_all)

    with gr.Tab("📊 Step 8: Model Evaluation (Metrics & Reports)"):
        gr.Markdown("### Test Accuracy, Macro Precision / Recall / F1, and Weighted F1")
        gr.Dataframe(value=get_comparison_table, label="Model Metrics Summary Table")
        gr.Markdown("### Full Evaluation JSON (Classification Metrics)")
        gr.Code(value=get_classification_reports, language="json", label="Classification Metrics Output")

    with gr.Tab("📈 Step 9: Model Comparison & Analysis"):
        gr.Markdown("### Model Comparison Chart (Accuracy & Macro-F1)")
        if os.path.exists("plots/model_comparison.png"):
            gr.Image("plots/model_comparison.png", label="Accuracy & Macro F1 Bar Chart")
        else:
            gr.Markdown("*Comparison chart will show after running 9-compare.py*")
            
        gr.Markdown("""
        ### Key Technical Discussion:
        - **Speed:** Linear models (SVM and Naive Bayes) train in under **1 second** over sparse TF-IDF matrices, whereas RNN/LSTM/GRU take multiple minutes due to sequential recurrence.
        - **Class Imbalance:** Stratified sampling maintained balanced class distributions, ensuring Macro-F1 and Weighted-F1 are aligned without bias toward either class.
        - **Architecture Trade-offs:**
          - **Simple RNN:** Struggles with long-range sequences due to vanishing gradients across 200 time steps.
          - **LSTM:** Utilizes cell state highways and 3 gates (forget, input, output) to preserve sentiment across long documents.
          - **GRU:** Simplifies LSTM into 2 gates (update, reset), delivering comparable accuracy with fewer parameters.
        """)
        
    with gr.Tab("📉 Deep Learning Learning Curves"):
        gr.Markdown("### Training vs. Validation Loss & Accuracy per Epoch")
        with gr.Row():
            if os.path.exists("plots/rnn_learning_curves.png"):
                gr.Image("plots/rnn_learning_curves.png", label="Simple RNN Curves")
            if os.path.exists("plots/lstm_learning_curves.png"):
                gr.Image("plots/lstm_learning_curves.png", label="LSTM Curves")
            if os.path.exists("plots/gru_learning_curves.png"):
                gr.Image("plots/gru_learning_curves.png", label="GRU Curves")

if __name__ == "__main__":
    demo.launch()
