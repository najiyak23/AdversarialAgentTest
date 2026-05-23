import pandas as pd

# load original labeled dataset
df = pd.read_csv("data/base_data.csv")

# keep only needed columns
df = df[["headline", "sentiment", "label"]]

# make copy
adv_df = df.copy()

# ------------------------
# Stress Test 1: flip sentiment
# ------------------------
adv_df["adversarial_sentiment"] = adv_df["sentiment"].replace({
    "positive": "negative",
    "negative": "positive",
    "neutral": "positive"
})

# ------------------------
# Stress Test 2: noisy headline
# ------------------------
adv_df["adversarial_headline"] = (
    "URGENT!!! " + adv_df["headline"] + " ???"
)

# save
adv_df.to_csv("data/adversarial_data.csv", index=False)

print("Adversarial dataset created successfully!")
print(adv_df.head())