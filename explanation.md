# NLP Task 2 — Project Explanation

**Goal:** Train and fairly compare five different models (SVM, Naive Bayes, Simple RNN, LSTM, GRU) on the same dataset to classify text as **Positive** or **Negative** sentiment.

---

## What is the Dataset?

We use the preprocessed IMDB movie review dataset from **Task 1** (`imdb_preprocessed.csv`). Each row is a cleaned movie review paired with its sentiment label (positive or negative). We use the `final_text` column as input and `sentiment` as the label to predict.

---

## The Pipeline — Step by Step

### Step 1 — `1-prepare-dataset.py` · Prepare the Shared Dataset

This is about getting the data ready before any model sees it.

**What we do, line by line:**

```python
df = pd.read_csv(r"...\imdb_preprocessed.csv")
```
Load the already-cleaned CSV from Task 1.

```python
df = df.dropna().drop_duplicates().reset_index(drop=True)
```
Remove any rows that are blank or exact copies of another row. This avoids feeding garbage or repeated data to our models.

```python
df = df[df["final_text"].str.strip() != ""].reset_index(drop=True)
```
Remove rows where the review text is empty (just spaces). An empty review would confuse every model.

```python
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
```
Turn the words "positive" and "negative" into numbers. Models work with numbers, not words, so **positive = 1** and **negative = 0**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    df["final_text"], df["label"], test_size=0.2, random_state=3, stratify=df["label"]
)
```
Split the data: **80% for training**, **20% for testing**. `random_state=3` means the split is always the same every time we run it. `stratify=df["label"]` means we make sure both halves have roughly the same ratio of positive-to-negative reviews — so the test set isn't accidentally all-positive or all-negative.

> [!IMPORTANT]
> This split is done **once** and saved to CSV files. All five models use the **exact same test rows** so comparisons are fair.

---

### Step 2 — `2-prepare-features.py` · Prepare Text Features

Models cannot read raw words directly. We need to convert text into numbers in two different ways — one for the simple models, one for the deep learning models.

**Why two formats?**
- SVM and Naive Bayes work best with a fixed-size number table (one number per word across the whole vocabulary).
- RNN, LSTM, GRU need text as a sequence of word IDs because they read word-by-word in order.

---

#### TF-IDF (for SVM and Naive Bayes)

```python
tfidf = TfidfVectorizer(max_features=20000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
```

**TF-IDF** stands for *Term Frequency – Inverse Document Frequency*. Plain language: it gives each word a score that reflects how important that word is in a specific review compared to all other reviews.

- A word like "the" appears in every review → low score (not useful for telling positive from negative).
- A word like "masterpiece" appears rarely and mainly in positive reviews → high score (very useful).

`fit_transform` on training: learns which 20,000 words to care about, then scores them.  
`transform` on test: applies the same scoring rules learned from training — no peeking at test data.

The result is a **matrix**: each row = one review, each column = one word's TF-IDF score.

---

#### Tokenizer + Padding (for RNN, LSTM, GRU)

```python
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200, padding="pre", truncating="pre")
X_test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=200, padding="pre", truncating="pre")
```

1. **Tokenizer** builds a dictionary: every word gets an ID number. `<OOV>` = any word not in the top 20,000 (Out Of Vocabulary).
2. `texts_to_sequences` converts each review from words → list of ID numbers.
3. `pad_sequences` makes every review exactly **200 numbers long**:
   - Short reviews get zeros added at the front (`padding="pre"`).
   - Long reviews get cut from the front (`truncating="pre"`), keeping the most recent words.

This uniform length is required because deep learning models need fixed-size input batches.

> [!NOTE]
> The tokenizer is **fit only on training data** and saved. The same tokenizer is reused for predictions on new text — this is critical. If you use a different tokenizer, the word IDs will mean different things and the model will produce garbage.

---

### Step 3 — `3-svm.py` · Support Vector Machine

```python
model = LinearSVC(max_iter=2000)
model.fit(X_train_tfidf, y_train)
```

**How SVM works (simple):** Imagine plotting every review as a point in space, with positive reviews on one side and negative on the other. SVM finds the best straight dividing line (called a hyperplane) that separates them with the biggest gap. New reviews are classified based on which side of that line they fall on.

`LinearSVC` is a fast version of SVM that works well with large text matrices like TF-IDF.

**Strength:** Very fast (trains in under 1 second). Performs surprisingly well on text.  
**Weakness:** Has no concept of word order — "not good" and "good not" look the same to it.

---

### Step 4 — `4-naive-bayes.py` · Naive Bayes

```python
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
```

**How Naive Bayes works (simple):** It learns the probability of each word appearing in positive vs. negative reviews. When given a new review, it multiplies the probabilities of all its words together and picks the class with the higher overall probability. It's called "naive" because it assumes each word is independent of the others — which isn't quite true in real language, but works well in practice.

`MultinomialNB` is designed for count-based features like TF-IDF.

**Strength:** Fastest training of all five models (under 0.1 seconds). Easy to understand.  
**Weakness:** Ignores word order completely. The "naive" independence assumption limits accuracy.

---

### Step 5 — `5-rnn.py` · Simple Recurrent Neural Network

```python
model = Sequential([
    Embedding(VOCAB_SIZE, 64, mask_zero=True),
    SimpleRNN(64),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
```

**How a Simple RNN works:** Unlike SVM and Naive Bayes, the RNN reads the review **word by word**, in order. It maintains a small "memory" that gets updated at each word. By the end of the review, that memory is supposed to capture the overall meaning.

**Layer by layer:**

| Layer | What it does |
|-------|-------------|
| `Embedding(20000, 64)` | Converts each word ID into a 64-number vector that captures meaning. Words with similar meanings end up close together. `mask_zero=True` tells the model to ignore the padding zeros. |
| `SimpleRNN(64)` | Reads the sequence step by step. At each step, it updates its hidden state based on the current word and the previous state. |
| `Dropout(0.5)` | Randomly turns off 50% of neurons during training. This forces the model not to rely too heavily on any single neuron, reducing overfitting. |
| `Dense(1, sigmoid)` | Final output: one number between 0 and 1. Above 0.5 = Positive, below = Negative. |

**Weakness:** Simple RNN has a **vanishing gradient** problem. When the review is long (200 words), information from the first words gets "forgotten" by the time it reaches the end. This limits how well it handles sentiment that spans the full review.

---

### Step 6 — `6-lstm.py` · Long Short-Term Memory

```python
model = Sequential([
    Embedding(VOCAB_SIZE, 64, mask_zero=True),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
```

Same structure as the RNN, but `SimpleRNN` is replaced with `LSTM`.

**How LSTM differs:** LSTM adds a **cell state** — a separate memory lane that runs alongside the main computation. Three gates control what information flows in and out:

| Gate | What it does |
|------|-------------|
| **Forget gate** | Decides what old information to throw away |
| **Input gate** | Decides what new information from the current word to remember |
| **Output gate** | Decides what part of the memory to send forward |

This gating mechanism lets the LSTM remember important things from early in the review (like "not" before "good") all the way until the end. It directly fixes the vanishing gradient problem of Simple RNN.

**Strength:** Better at long-range dependencies in text.  
**Cost:** More parameters = slower to train than Simple RNN.

---

### Step 7 — `7-gru.py` · Gated Recurrent Unit

```python
model = Sequential([
    Embedding(VOCAB_SIZE, 64, mask_zero=True),
    GRU(64),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
```

**How GRU differs from LSTM:** GRU is a simplified version of LSTM. It merges the forget and input gates into one **update gate**, and uses a **reset gate** instead of a separate output gate. This means fewer parameters and faster training, while still solving the vanishing gradient problem.

| Feature | Simple RNN | LSTM | GRU |
|---------|-----------|------|-----|
| Memory gates | None | 3 gates | 2 gates |
| Handles long text | Poorly | Well | Well |
| Training speed | Medium | Slowest | Faster than LSTM |
| Parameters | Fewest | Most | Between RNN and LSTM |

In practice, GRU and LSTM perform similarly on most text tasks. GRU is often the better choice when training time matters.

---

### Training Settings (applies to RNN, LSTM, GRU)

```python
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001, clipnorm=1.0), metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, callbacks=[early_stop])
```

- **binary_crossentropy loss:** Measures how wrong the prediction is for a yes/no problem. Lower = better.
- **Adam optimizer:** Adjusts the model's internal numbers to minimize loss. `clipnorm=1.0` prevents any single update from being too large (gradient clipping).
- **EarlyStopping:** If validation loss stops improving for 1 epoch in a row, training stops and the best weights are restored. This prevents overfitting.
- **validation_split=0.1:** 10% of training data is held back to monitor training progress each epoch.
- **batch_size=64:** Process 64 reviews at a time before updating weights.

---

### Step 8 — `8-evaluate.py` · Evaluate Every Model

This script loads all five trained models and tests them on the held-out test set.

#### Metrics explained

**Accuracy**
> Out of all predictions, what fraction was correct?

$$\text{Accuracy} = \frac{\text{Correct predictions}}{\text{Total predictions}}$$

Simple but can be misleading if one class is much more common than the other.

---

**Precision** (per class)
> Of all the times we predicted "Positive", how often were we actually right?

High precision means: when the model says positive, it's usually correct. Few false alarms.

---

**Recall** (per class)
> Of all the actual "Positive" reviews, how many did we correctly catch?

High recall means: the model finds most of the positive reviews. Few misses.

---

**F1 Score**
> The balance between Precision and Recall. Harmonic mean of both.

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

A single number that tells you how well the model does on a class when both precision and recall matter.

---

**Macro F1**
> Average F1 across all classes, treating each class equally regardless of size.

We use this as the primary ranking metric because it doesn't let a large class hide poor performance on a smaller one.

**Weighted F1**
> Average F1 weighted by how many samples each class has.

Tells you overall performance accounting for class frequency.

---

**Confusion Matrix**

A 2×2 table that shows:

|  | Predicted Negative | Predicted Positive |
|--|---|---|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

Reading it: the diagonal (top-left, bottom-right) = correct predictions. Off-diagonal = mistakes.

---

**Training Time**
Recorded in seconds. Shows the cost of training each model. Important when deciding which model to deploy in a real system.

---

#### Learning Curves (DL models only)

For RNN, LSTM, and GRU we also plot **accuracy and loss per epoch** for both training and validation sets:

- **Train Acc / Train Loss:** How well the model learned from training data.
- **Val Acc / Val Loss:** How well it generalizes to data it hasn't seen during training.

A healthy curve: training and validation both improve together. If training improves but validation gets worse, the model is **overfitting** (memorizing instead of learning).

---

### Step 9 — `9-compare.py` · Compare All Five Models

```python
df_compare = pd.DataFrame(table_data)
```

Builds a comparison table with all five models, their metrics, and training times side by side. Also generates a bar chart (`plots/model_comparison.png`) showing Accuracy and Macro-F1 for each model.

**Key discussion points from the results:**

- **Speed:** SVM and Naive Bayes train in under 1 second. RNN, LSTM, GRU take minutes because they process each word sequentially and have many more parameters to update.
- **Class Imbalance:** Stratified sampling kept positive/negative balanced in both train and test sets. This is why Macro-F1 and Weighted-F1 are close together — there's no major class imbalance pulling the numbers apart.
- **Architecture Trade-offs:**
  - **Simple RNN:** Reasonable accuracy but limited by forgetting early words in long reviews.
  - **LSTM:** Handles long context well through its gating system.
  - **GRU:** Similar accuracy to LSTM, slightly fewer parameters, often trains faster.
- **Best model** is chosen by **Macro F1**, not accuracy alone — because F1 reflects how balanced the model is across both classes.

---

### Step 10 — `10-predict.py` · Predict Unseen Text

Tests all five models on five brand-new reviews that were never in the training or test set.

**The prediction pipeline for each review:**

```python
cleaned = clean_text(raw)                          # Same preprocessing as training
feat_tfidf = tfidf.transform([cleaned])            # For SVM and NB
feat_pad = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=200, ...)  # For DL models
```

> [!IMPORTANT]
> The preprocessing and feature extraction **must exactly match** what was used during training. If you use a different tokenizer or TF-IDF, the word IDs and scores mean different things, and predictions will be wrong. This is why we saved and reloaded the same `tfidf.pkl` and `tokenizer.pkl`.

Output: a table showing what each model predicts for each review.

---

## Interface — `interface/app.py`

A Gradio web dashboard with five tabs:

| Tab | What it shows |
|-----|--------------|
| **Predict with Specific Model** | Type any text, pick one model, get its prediction and confidence % |
| **Multi-Model Comparative Prediction** | Type any text, all 5 models predict at once — side by side table |
| **Step 8: Model Evaluation** | Metrics table (accuracy, macro F1, weighted F1, etc.) + full JSON classification report |
| **Step 9: Model Comparison** | Bar chart of accuracy vs macro-F1, plus written analysis of tradeoffs |
| **Deep Learning Learning Curves** | The accuracy/loss plots for RNN, LSTM, and GRU across training epochs |

**Confidence % for SVM:** Since LinearSVC doesn't output probabilities directly, we apply a sigmoid function to its raw decision score to convert it into a 0–100% range (Platt scaling).

**Confidence % for Naive Bayes:** Directly from `predict_proba()`.

**Confidence % for DL models:** The final sigmoid layer already outputs a value between 0 and 1 (probability of positive). We show that directly.

---

## How Each Model Compares at a Glance

| Model | Input Format | Reads Order? | Handles Long Text? | Speed |
|-------|-------------|-------------|-------------------|-------|
| SVM | TF-IDF | ✗ No | ✗ No | ⚡ Very Fast |
| Naive Bayes | TF-IDF | ✗ No | ✗ No | ⚡ Fastest |
| Simple RNN | Padded Sequences | ✓ Yes | ⚠ Struggles | 🐢 Slow |
| LSTM | Padded Sequences | ✓ Yes | ✓ Yes (3 gates) | 🐢 Slowest |
| GRU | Padded Sequences | ✓ Yes | ✓ Yes (2 gates) | 🐢 Slow |

---

## File Structure

```
NLP_task2/
├── 1-prepare-dataset.py      # Load, clean, split data
├── 2-prepare-features.py     # TF-IDF + tokenizer/padding
├── 3-svm.py                  # Train LinearSVC
├── 4-naive-bayes.py          # Train MultinomialNB
├── 5-rnn.py                  # Train Simple RNN
├── 6-lstm.py                 # Train LSTM
├── 7-gru.py                  # Train GRU
├── 8-evaluate.py             # Evaluate all models, generate plots
├── 9-compare.py              # Comparison table + bar chart
├── 10-predict.py             # Predict 5 unseen texts with all models
├── explanation.md            # This file
├── data/                     # Saved splits, features, results, histories
├── models/                   # Saved trained model files
├── plots/                    # Generated PNG charts
└── interface/
    └── app.py                # Gradio web dashboard
```
