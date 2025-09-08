from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os, datetime, random
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

# ========================
# Load TensorFlow model & preprocessors
# ========================
model = load_model("chatbot_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("response_map.pkl", "rb") as f:
    response_map = pickle.load(f)   # intent → bot_response

max_len = 25  # must match training script

# ========================
# Incremental learning model (hybrid)
# ========================
vectorizer = HashingVectorizer(n_features=2**16)
clf = SGDClassifier(loss="log_loss")

# Initialize with intents from response_map
intents = list(response_map.keys())
clf.partial_fit(vectorizer.transform(intents), intents, classes=np.unique(intents))

# ========================
# Logging
# ========================
LOG_FILE = "chat_logs.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "user_message", "predicted_intent", "bot_response", "model_source"]).to_csv(LOG_FILE, index=False)

def log_conversation(user_message, predicted_intent, bot_reply, source="keras"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df = pd.DataFrame([[timestamp, user_message, predicted_intent, bot_reply, source]],
                          columns=["timestamp", "user_message", "predicted_intent", "bot_response", "model_source"])
    log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)

# Reload logs into incremental model
try:
    log_data = pd.read_csv(LOG_FILE)
    if not log_data.empty:
        X_logs = vectorizer.transform(log_data["user_message"].astype(str).tolist())
        y_logs = log_data["predicted_intent"].astype(str).tolist()
        if len(set(y_logs)) > 0:
            clf.partial_fit(X_logs, y_logs, classes=np.unique(y_logs))
except Exception as e:
    print("Log reload skipped:", e)

# ========================
# Flask
# ========================
app = Flask(__name__)
CORS(app)

# Templates for dynamic rephrasing
templates = [
    "Here’s what I found: {answer}",
    "Good question! {answer}",
    "Sure thing 👍 {answer}",
    "Here’s the info you need: {answer}",
    "Absolutely! {answer}",
    "{answer} (hope that clears things up!)",
    "No worries — {answer}",
]

last_responses = {}

def ensure_minimum_words(text, min_words=40):
    """Pad response to at least 40 words by elaborating naturally."""
    words = text.split()
    if len(words) < min_words:
        filler = (
            " To provide a bit more detail, our council is always working on multiple "
            "programs and activities that benefit not only the youth but also the wider "
            "community. We encourage participation and feedback to continuously improve "
            "our initiatives for everyone in Brgy. Buhangin."
        )
        text = text + filler
    return text

def generate_dynamic_reply(base_reply, intent):
    """Wrap base reply into a random template and ensure minimum length."""
    chosen_template = random.choice(templates)
    reply = chosen_template.format(answer=base_reply)

    # Avoid repeating the same reply for the same intent
    if intent in last_responses and last_responses[intent] == reply:
        alt_templates = [t for t in templates if t.format(answer=base_reply) != reply]
        if alt_templates:
            reply = random.choice(alt_templates).format(answer=base_reply)

    reply = ensure_minimum_words(reply, 40)
    last_responses[intent] = reply
    return reply

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get("message", "").lower()

    # ---------- Step 1: TensorFlow model prediction ----------
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")

    pred = model.predict(padded)
    intent_idx = np.argmax(pred)
    predicted_intent = label_encoder.inverse_transform([intent_idx])[0]
    confidence = float(np.max(pred))

    # ---------- Step 2: Incremental model backup ----------
    X = vectorizer.transform([message])
    if hasattr(clf, "classes_"):
        alt_intent = clf.predict(X)[0]
    else:
        alt_intent = predicted_intent

    # ---------- Step 3: Choose response ----------
    bot_reply = "I'm not sure how to respond yet."
    source = "keras"

    if confidence < 0.6:  # low confidence → fallback
        predicted_intent = alt_intent
        source = "incremental"

    if predicted_intent in response_map:
        base_reply = response_map[predicted_intent]
        bot_reply = generate_dynamic_reply(base_reply, predicted_intent)

    # ---------- Step 4: Log ----------
    log_conversation(message, predicted_intent, bot_reply, source)

    return jsonify({
        "intent": predicted_intent,
        "confidence": confidence,
        "response": bot_reply,
        "source": source
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    """Update incremental model with corrected intent"""
    data = request.json
    message = data["message"].lower()
    correct_intent = data["correct_intent"]

    X_new = vectorizer.transform([message])
    clf.partial_fit(X_new, [correct_intent])

    # Log corrected entry
    log_conversation(message, correct_intent, "Corrected by user", source="feedback")

    return jsonify({"status": "updated", "new_intent": correct_intent})

# ========================
# Run
# ========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
