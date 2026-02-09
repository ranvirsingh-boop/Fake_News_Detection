import streamlit as st
import joblib
import re
from nltk.stem import PorterStemmer

# Load trained model and vectorizer


model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")



# Hardcoded stopwords (NO nltk.download, NO stopwords.words)
stop_words = {
    'a','an','the','and','or','but','if','while','with','without','of','at','by',
    'for','to','from','in','on','is','are','was','were','be','been','being',
    'this','that','these','those','it','its','as','about','into','than','then',
    'so','such','too','very','can','will','just','not'
}

ps = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="TruthLens - Fake News Detection", layout="centered")

st.title("📰 TruthLens: Fake News Detection System")
st.write("Enter a news **headline/title** to check whether it is **Fake or Real**.")

news_input = st.text_area("Enter News Title", height=150)

if st.button("Detect"):
    if news_input.strip() == "":
        st.warning("Please enter a news title.")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max() * 100

        if prediction == 1:
            st.success(f"✅ Real News ({confidence:.2f}% confidence)")
        else:
            st.error(f"❌ Fake News ({confidence:.2f}% confidence)")

