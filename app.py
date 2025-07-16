from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

MODEL_PATH = "disaster_tweet_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 100

# Load model and tokenizer
if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError("Model or tokenizer file missing")

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""
    confidence = ""

    if request.method == "POST":
        tweet = request.form.get("tweet", "")
        if tweet.strip() == "":
            prediction_text = "⚠️ Please enter a tweet."
        else:
            sequence = tokenizer.texts_to_sequences([tweet])
            padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
            prediction = model.predict(padded)[0][0]
            label = "🚨 Disaster Tweet" if prediction > 0.5 else "✅ Not a Disaster Tweet"
            confidence = f"{prediction:.2f}"
            prediction_text = label

    return render_template("index.html", prediction_text=prediction_text, confidence=confidence)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
