import numpy as np
import pandas as pd
import pickle
import os
import getEmbeddings
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from keras.preprocessing import sequence
from collections import Counter

# --- কনফিগারেশন ---
VOCAB_SIZE = 20000
MAX_LEN = 500

print("⏳ Loading Data for LSTM...")
if not os.path.exists('datasets/train.csv'):
    print("❌ Error: 'datasets/train.csv' not found!")
    exit()

data = pd.read_csv('datasets/train.csv')
data.dropna(subset=['text', 'label'], inplace=True)
data = data.reset_index(drop=True)

# টেক্সট ক্লিনিং
print("⏳ Cleaning Text (This takes a moment)...")
data['clean_text'] = data['text'].apply(lambda x: getEmbeddings.cleanup(x) if isinstance(x, str) else "")
labels = data['label'].values

# টোকেনাইজার তৈরি
print("⏳ Creating Tokenizer...")
all_words = []
for text in data['clean_text']:
    all_words.extend(text.split())

count = Counter(all_words)
word_bank = {w: i+1 for i, (w, _) in enumerate(count.most_common(VOCAB_SIZE))}

# টোকেনাইজার সেভ
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(word_bank, f)
print("✅ Tokenizer Saved!")

# সিকোয়েন্স তৈরি
print("⏳ Processing Sequences...")
X_seq = []
for text in data['clean_text']:
    seq = [word_bank[word] for word in text.split() if word in word_bank]
    X_seq.append(seq)

X_pad = sequence.pad_sequences(X_seq, maxlen=MAX_LEN)

# LSTM মডেল তৈরি এবং ট্রেনিং
print("🚀 Starting Bi-LSTM Training (You will see a progress bar now)...")

lstm_model = Sequential([
    Embedding(VOCAB_SIZE+1, 128, input_length=MAX_LEN),
    Bidirectional(LSTM(128)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# verbose=1 দেওয়া হয়েছে যাতে প্রোগ্রেস বার দেখা যায়
lstm_model.fit(X_pad, labels, epochs=10, batch_size=64, verbose=1)

lstm_model.save("lstm_model.h5")
print("\n🎉 Bi-LSTM Model Saved Successfully!")
print("✅ All models are ready. Now run 'streamlit run app.py'")