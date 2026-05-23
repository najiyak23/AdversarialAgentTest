import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("results/adversarial_predictions.csv")

# remove failed rows if any
df = df[df["llm_prediction"] != "ERROR"]

# metrics
acc = accuracy_score(df["label"], df["llm_prediction"])
report = classification_report(df["label"], df["llm_prediction"], output_dict=True)
cm = confusion_matrix(df["label"], df["llm_prediction"])

# print
print("Adversarial Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(df["label"], df["llm_prediction"]))
print("\nConfusion Matrix:\n")
print(cm)

# save classification report
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("results/classification_report.csv")

# save confusion matrix
cm_df = pd.DataFrame(
    cm,
    index=["BUY", "HOLD", "SELL"],
    columns=["BUY", "HOLD", "SELL"]
)
cm_df.to_csv("results/confusion_matrix.csv")

# save summary text
with open("results/evaluation_summary.txt", "w") as f:
    f.write(f"Adversarial Accuracy: {acc}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(df["label"], df["llm_prediction"]))
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\nSaved in results/ folder")