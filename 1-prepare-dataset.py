import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\Administrator\Desktop\NLP_task\imdb_preprocessed.csv")
df = df.dropna().drop_duplicates().reset_index(drop=True)
df = df[df["final_text"].str.strip() != ""].reset_index(drop=True)
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

X_train, X_test, y_train, y_test = train_test_split(
    df["final_text"], df["label"], test_size=0.2, random_state=3, stratify=df["label"]
)

X_train.reset_index(drop=True).to_csv("data/X_train.csv", index=False)
X_test.reset_index(drop=True).to_csv("data/X_test.csv", index=False)
y_train.reset_index(drop=True).to_csv("data/y_train.csv", index=False)
y_test.reset_index(drop=True).to_csv("data/y_test.csv", index=False)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Class distribution (train):\n{y_train.value_counts()}")
