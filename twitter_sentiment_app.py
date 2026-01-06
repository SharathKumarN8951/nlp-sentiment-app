import streamlit as st
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# CONSTANTS (same as training)
# -------------------------------
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.joblib"

MAX_SEQUENCE_LENGTH = 40

LABELS = ["Negative", "Neutral", "Positive"]

# -------------------------------
# LOAD MODEL & TOKENIZER
# -------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_sentiment(text):
    # Convert text to sequence
    sequences = tokenizer.texts_to_sequences([text])

    # Pad sequence
    padded = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    # Predict
    prediction = model.predict(padded)

    # Get label index
    label_index = prediction.argmax(axis=1)[0]

    return LABELS[label_index], prediction[0]

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("🧠 Sentiment Analysis App")
st.write("Enter a sentence and predict its sentiment.")

user_input = st.text_area("✍️ Enter your text here:")

if st.button("🔍 Predict Sentiment"):
    if user_input.strip():
        sentiment, scores = predict_sentiment(user_input)

        st.success(f"**Predicted Sentiment:** {sentiment}")

        st.subheader("📊 Confidence Scores")
        st.write({
            "Negative": float(scores[0]),
            "Neutral": float(scores[1]),
            "Positive": float(scores[2])
        })
    else:
        st.warning("⚠️ Please enter some text.")
