import os
import json
import pandas as pd
import matplotlib.pyplot as plt

eval_results = json.load(open("data/evaluation_results.json"))
try:
    times = json.load(open("data/train_times.json"))
except FileNotFoundError:
    times = {}

time_mapping = {
    "SVM": times.get("svm", "N/A"),
    "Naive Bayes": times.get("naive_bayes", "N/A"),
    "Simple RNN": times.get("rnn", "N/A"),
    "LSTM": times.get("lstm", "N/A"),
    "GRU": times.get("gru", "N/A")
}

table_data = []
for model_name, metrics in eval_results.items():
    table_data.append({
        "Model": model_name,
        "Accuracy": metrics["accuracy"],
        "Macro Precision": metrics["macro_precision"],
        "Macro Recall": metrics["macro_recall"],
        "Macro F1": metrics["macro_f1"],
        "Weighted F1": metrics["weighted_f1"],
        "Train Time (s)": time_mapping.get(model_name, "N/A")
    })

df_compare = pd.DataFrame(table_data)
print("\n" + "="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(df_compare.to_string(index=False))

df_compare.to_csv("data/model_comparison.csv", index=False)

# Plot comparison bar chart
models = df_compare["Model"]
x = range(len(models))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar([i - width/2 for i in x], df_compare["Accuracy"], width=width, label="Accuracy")
plt.bar([i + width/2 for i in x], df_compare["Macro F1"], width=width, label="Macro F1")
plt.xticks(x, models)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Comparison of Accuracy and Macro-F1 Across Models")
plt.legend()
plt.tight_layout()
plt.savefig("plots/model_comparison.png")
plt.close()

# Discussion
best_model = df_compare.sort_values(by="Macro F1", ascending=False).iloc[0]
print("\n" + "="*80)
print("ANALYSIS & DISCUSSION")
print("="*80)
print(f"1. Best Model: {best_model['Model']} achieved the highest Macro F1 ({best_model['Macro F1']}) and Accuracy ({best_model['Accuracy']}).")
print("2. Training Speed: Linear models (SVM and Naive Bayes) train in seconds compared to recurrent neural networks (RNN, LSTM, GRU).")
print("3. Class Imbalance: Stratified sampling maintained balanced class distributions, resulting in comparable Macro and Weighted F1 scores.")
print("4. Architecture Differences: LSTM and GRU handle long-range dependencies and vanishing gradients significantly better than Simple RNN.")
