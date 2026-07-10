import streamlit as st
import numpy as np
import pickle
import getEmbeddings  # আপনার টেক্সট ক্লিনিং ফাংশন
from gensim.models import Doc2Vec
from keras.models import load_model
from keras.preprocessing import sequence
import os

# --- কনফিগারেশন ---
MAX_LEN = 500

st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="wide")

# --- DEVELOPER NAME (TOP HEADING) ---
st.markdown("""
    <style>
    .dev-name {
        font-size: 20px;
        color: #4CAF50; /* সবুজ রং */
        font-weight: bold;
        text-align: center;
        margin-bottom: 0px;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #333;
        margin-top: -10px;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 10px;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .real { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .fake { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    </style>
    """, unsafe_allow_html=True)

# --- নাম প্রদর্শন ---
st.markdown('<div class="dev-name">Developed by Golam Rohman</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">🕵️ Fake News Detector System</div>', unsafe_allow_html=True)
st.markdown("---")


# --- রিসোর্স লোডিং ---
@st.cache_resource
def load_resources():
    try:
        d2v = Doc2Vec.load("d2v_model.doc2vec")
        with open("svm_model.pkl", "rb") as f:
            svm = pickle.load(f)
        with open("nb_model.pkl", "rb") as f:
            nb = pickle.load(f)
        keras_nn = load_model("keras_model.h5")
        lstm = load_model("lstm_model.h5")
        with open("tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)
        return d2v, svm, nb, keras_nn, lstm, tokenizer
    except Exception as e:
        return None, None, None, None, None, None


# লোডিং
d2v_model, svm_model, nb_model, keras_model, lstm_model, word_bank = load_resources()

if not d2v_model:
    st.error("❌ Error: Could not load models. Please make sure all .pkl and .h5 files exist.")
    st.stop()

# --- সাইডবার সেটিংস ---
st.sidebar.title("⚙️ Calibration")
model_choice = st.sidebar.selectbox("Select Model", ["Bi-LSTM (Best)", "Keras Neural Net", "SVM", "Naive Bayes"])

st.sidebar.markdown("---")
st.sidebar.write("### 🔧 Result Fixer")
st.sidebar.info("If the result looks wrong (e.g. Real news marked as Fake), use this checkbox to flip the logic.")
invert_logic = st.sidebar.checkbox("🔄 Invert/Flip Result Logic", value=False)

# --- মেইন উইন্ডো ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📰 Input News")
    news_text = st.text_area("Paste the full article content here:", height=300)
    analyze_btn = st.button("🔍 Analyze Veracity", type="primary")

with col2:
    st.subheader("🧠 Diagnostics")

    if analyze_btn and news_text:
        # 1. Cleaning
        cleaned_text = getEmbeddings.cleanup(news_text)

        # 2. Prediction Logic
        raw_score = 0.0  # 0 (Real) to 1 (Fake)

        if model_choice == "Bi-LSTM (Best)":
            words = cleaned_text.split()
            known_words = [w for w in words if w in word_bank]

            st.write(f"**Total Words:** {len(words)}")
            st.write(f"**Known Words:** {len(known_words)}")

            if len(known_words) < 3:
                st.warning("⚠️ The model recognizes too few words! It may give random results.")

            seq = [word_bank[word] for word in words if word in word_bank]
            padded = sequence.pad_sequences([seq], maxlen=MAX_LEN)
            raw_score = float(lstm_model.predict(padded)[0][0])

        else:  # Vector models
            vector = d2v_model.infer_vector(cleaned_text.split()).reshape(1, -1)
            if model_choice == "SVM":
                pred = svm_model.predict(vector)[0]
                raw_score = 1.0 if pred == 1 else 0.0
            elif model_choice == "Naive Bayes":
                prob = nb_model.predict_proba(vector)[0]
                raw_score = prob[1]
            elif model_choice == "Keras Neural Net":
                raw_score = float(keras_model.predict(vector)[0][0])

        # 3. Decision Logic
        is_fake = raw_score > 0.5
        if invert_logic:
            is_fake = not is_fake

        confidence = raw_score if is_fake else (1 - raw_score)
        if invert_logic: confidence = raw_score if not is_fake else (1 - raw_score)

        # 4. Result Display
        label = "FAKE NEWS" if is_fake else "REAL NEWS"

        st.markdown("---")
        if label == "REAL NEWS":
            st.markdown(f'<div class="result-card real">✅ {label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card fake">❌ {label}</div>', unsafe_allow_html=True)

        st.write("")
        st.metric("Confidence Score", f"{confidence * 100:.2f}%")

        with st.expander("Debug Info"):
            st.write(f"Raw Score: {raw_score:.4f}")
            st.write(f"Invert Active: {invert_logic}")

    elif analyze_btn:
        st.warning("Please enter text first.")