from google_verify import google_search, extract_evidence, verify_claim

import streamlit as st
import re
import pandas as pd
import joblib
import nltk

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 AI-Based Fake News Detection System")
st.write("Detect fake news using AI and verify claims using trusted sources.")

# ---------- NLTK SETUP ----------
@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")

load_nltk()

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ---------- TEXT CLEANING ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------- USER INPUT ----------
user_input = st.text_area("Enter News Title or Content")

# ---------- BUTTON ----------
if st.button("Detect"):

    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned = clean_text(user_input)
        X = vectorizer.transform([cleaned])

        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X).max() * 100

        # ---------- ML RESULT ----------
        if prediction == 1:
            st.success(f"✅ Real News ({confidence:.2f}% confidence)")
        else:
            st.error(f"❌ Fake News ({confidence:.2f}% confidence)")

        # ---------- UNCERTAINTY HANDLING ----------
        if confidence < 85:
            st.warning("⚠️ Low confidence prediction — verifying using trusted sources.")

            # ---------- GOOGLE VERIFICATION ----------
            from google_verify import google_search, extract_evidence, verify_claim

            search_data = google_search(user_input)
            evidence = extract_evidence(search_data)
            verdict = verify_claim(user_input, evidence)

            st.subheader("🔎 Web Verification Result")
            st.info(verdict)

            if evidence:
                st.subheader("🧾 Evidence from Trusted Sources")
                for e in evidence[:3]:
                    st.write(f"• **{e['title']}**")
                    st.write(e["link"])
            else:
                st.info("No authoritative sources found.")

        else:
            st.info("ℹ️ High confidence — web verification not required.")

# ---------- FOOTER ----------
st.markdown("---")
st.caption(
    "Note: This system uses AI-based language analysis. "
    "For low-confidence cases, it performs web-based verification using trusted news sources."
)

