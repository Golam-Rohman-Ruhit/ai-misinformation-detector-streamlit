import numpy as np
import pandas as pd
import pickle
import os
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from keras.preprocessing import sequence
from keras.utils import to_categorical
from collections import Counter
import getEmbeddings  # আপনার টেক্সট ক্লিনিং ফাংশন ব্যবহারের জন্য

# --- কনফিগারেশন ---
VOCAB_SIZE = 20000
MAX_LEN = 500
EMBEDDING_DIM = 300

print("⏳ 1. Training Doc2Vec Model & Saving...")
# ডেটা লোড
data = pd.read_csv('datasets/train.csv')
data.dropna(subset=['text', 'label'], inplace=True)
data = data.reset_index(drop=True)

# টেক্সট ক্লিনিং
data['clean_text'] = data['text'].apply(lambda x: getEmbeddings.cleanup(x) if isinstance(x, str) else "")

# Doc2Vec ট্রেনিং
tagged_data = [TaggedDocument(words=row.split(), tags=[str(i)]) for i, row in enumerate(data['clean_text'])]
d2v_model = Doc2Vec(vector_size=EMBEDDING_DIM, min_count=2, epochs=20)
d2v_model.build_vocab(tagged_data)
d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
d2v_model.save("d2v_model.doc2vec") # Doc2Vec মডেল সেভ
print("✅ Doc2Vec Model Saved!")

# ভেক্টর তৈরি
vectors = np.array([d2v_model.infer_vector(row.split()) for row in data['clean_text']])
labels = data['label'].values

print("⏳ 2. Training & Saving SVM...")
svm_model = LinearSVC(max_iter=10000)
svm_model.fit(vectors, labels)
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)
print("✅ SVM Saved!")

print("⏳ 3. Training & Saving Naive Bayes...")
nb_model = GaussianNB()
nb_model.fit(vectors, labels)
with open("nb_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)
print("✅ Naive Bayes Saved!")

print("⏳ 4. Training & Saving Keras Neural Net...")
k_model = Sequential([
    Dense(256, input_dim=EMBEDDING_DIM, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') # Binary output
])
k_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
k_model.fit(vectors, labels, epochs=20, batch_size=64, verbose=0)
k_model.save("keras_model.h5")
print("✅ Keras NN Saved!")

print("⏳ 5. Training & Saving Bi-LSTM (Processing Text)...")
# LSTM এর জন্য টোকেনাইজার
all_words = []
for text in data['clean_text']:
    all_words.extend(text.split())

count = Counter(all_words)
word_bank = {w: i+1 for i, (w, _) in enumerate(count.most_common(VOCAB_SIZE))}

# টোকেনাইজার সেভ
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(word_bank, f)

# সিকোয়েন্স তৈরি
X_seq = []
for text in data['clean_text']:
    seq = [word_bank[word] for word in text.split() if word in word_bank]
    X_seq.append(seq)

X_pad = sequence.pad_sequences(X_seq, maxlen=MAX_LEN)

# LSTM মডেল
lstm_model = Sequential([
    Embedding(VOCAB_SIZE+1, 128, input_length=MAX_LEN),
    Bidirectional(LSTM(128)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_pad, labels, epochs=10, batch_size=64, verbose=0)
lstm_model.save("lstm_model.h5")
print("✅ Bi-LSTM Saved!")

print("\n🎉 All models saved successfully! Now run 'streamlit run app.py'")