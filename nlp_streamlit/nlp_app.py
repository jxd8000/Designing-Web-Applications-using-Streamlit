# nlp_App.py
# Create an NLP application that will predict whether the user's text input has a positive, negative, or neutral sentiment.

import streamlit as st
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob


try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def clean_text(text):
    # A. Keeping only Text and digits
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    # Removes Whitespaces
    text = re.sub(r"\s+", " ", text)
    # Removing Links if any
    text = re.sub(r"http\S+", " ", text)
    # Removes Punctuations and Numbers
    text = re.sub(r"\d+", " ", text)
    # Splitting Text
    words = text.split()
    # Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    text = " ".join(lemmatized_words)
    return text




st.title("NLP Sentiment App")
st.subheader("Type something below and click Analyze")

# Text input field (multiline)
user_input = st.text_area("Input text here:", height=200)


# Perform sentiment analysis when the button is clicked.
if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text first...")
    else:
        cleaned = clean_text(user_input)

        blob = TextBlob(cleaned)
        result = blob.sentiment.polarity   # [-1, 1]

        if result > 0:
            custom_emoji = ":blush:"
            st.success("Happy {}  (Polarity: {:.3f})".format(custom_emoji, result))
        elif result < 0:
            custom_emoji = ":disappointed:"
            st.warning("Sad {}  (Polarity: {:.3f})".format(custom_emoji, result))
        else:
            custom_emoji = ":confused:"
            st.info("Confused/Neutral {}  (Polarity: {:.3f})".format(custom_emoji, result))

        st.write("Polarity Score is:", result)
