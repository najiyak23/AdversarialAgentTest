import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# load env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# only first 20 rows
df = pd.read_csv("data/adversarial_data.csv").head(20)

predictions = []

def ask_llm(headline):
    prompt = f"""
    Read this cryptocurrency headline and output only one word:
    BUY, SELL, or HOLD.

    Headline: {headline}
    """

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip().upper()

        except Exception as e:
            print(f"Retry {attempt+1}: {e}")
            time.sleep(15)

    return "HOLD"


for i, headline in enumerate(df["adversarial_headline"]):
    print(f"Processing {i+1}/{len(df)}")
    pred = ask_llm(headline)
    predictions.append(pred)
    time.sleep(5)

df["llm_prediction"] = predictions
df.to_csv("results/adversarial_predictions.csv", index=False)

print("Done!")