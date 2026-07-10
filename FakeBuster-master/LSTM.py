#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection
High-Performance Bidirectional LSTM model
Target Accuracy: 95%+
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from keras.preprocessing import sequence
from collections import Counter
import getEmbeddings
import tensorflow as tf

# --- High-Performance Hyperparameters ---
top_words = 20000  # শব্দভাণ্ডার ৫০০০ থেকে বাড়িয়ে ২০০০০ করা হলো (Rare words ধরার জন্য)
epoch_num = 10  # ইপক ৫ থেকে বাড়িয়ে ১০ করা হলো (ভালোভাবে শেখার জন্য)
batch_size = 64
max_review_length = 500
embedding_vecor_length = 128  # ৩২ থেকে বাড়িয়ে ১২৮ করা হলো (গভীর অর্থ বোঝার জন্য)


def plot_cmat(yte, ypred):
    '''Plotting confusion matrix using seaborn'''
    cm = confusion_matrix(yte, ypred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake (0)', 'Real (1)'],
                yticklabels=['Fake (0)', 'Real (1)'])
    plt.title('Bi-LSTM Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


def run_lstm_model():
    print("WARNING: Overwriting existing .npy files with raw text strings for LSTM tokenization.")
    getEmbeddings.clean_data()

    # Load data
    xtr = np.load('./xtr_shuffled.npy', allow_pickle=True)
    xte = np.load('./xte_shuffled.npy', allow_pickle=True)
    y_train = np.load('./ytr_shuffled.npy')
    y_test = np.load('./yte_shuffled.npy')

    # --- Data Preprocessing ---
    cnt = Counter()
    x_train = []

    # Robust loading (fixing split error)
    for x in xtr:
        text = x.item() if isinstance(x, np.ndarray) and x.size == 1 else x
        if isinstance(text, str):
            words = text.split()
            x_train.append(words)
            for word in words:
                cnt[word] += 1

    most_common = cnt.most_common(top_words + 1)
    word_bank = {}
    id_num = 1
    for word, freq in most_common:
        word_bank[word] = id_num
        id_num += 1

    for news in x_train:
        i = 0
        while i < len(news):
            if news[i] in word_bank:
                news[i] = word_bank[news[i]]
                i += 1
            else:
                del news[i]

    y_train = list(y_train)

    # Filter short news
    i = 0
    while i < len(x_train):
        if len(x_train[i]) > 10:
            i += 1
        else:
            del x_train[i]
            del y_train[i]

    x_test = []
    for x in xte:
        text = x.item() if isinstance(x, np.ndarray) and x.size == 1 else x
        if isinstance(text, str):
            x_test.append(text.split())

    for news in x_test:
        i = 0
        while i < len(news):
            if news[i] in word_bank:
                news[i] = word_bank[news[i]]
                i += 1
            else:
                del news[i]

    X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # --- High-Performance Model Architecture ---
    model = Sequential()

    # 1. Bigger Embedding Layer
    model.add(Embedding(top_words + 2, embedding_vecor_length, input_length=max_review_length))

    # 2. Bidirectional LSTM (The Game Changer)
    # এটি কনটেক্সট দুই দিক থেকে শিখবে। 128 ইউনিট ব্যবহার করা হচ্ছে।
    model.add(Bidirectional(LSTM(128)))

    # 3. Dropout (To prevent overfitting)
    model.add(Dropout(0.3))

    # 4. Extra Dense Layer with ReLU
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    # 5. Output Layer
    model.add(Dense(1, activation='sigmoid'))

    # Adam optimizer with default learning rate usually works best for LSTMs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    print("\nTraining High-Performance Bi-LSTM model (10 Epochs)...")
    # validation_split ব্যবহার করা হচ্ছে যাতে ট্রেনিং চলাকালীনই ভ্যালিডেশন বোঝা যায়
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_num, batch_size=batch_size)

    # --- Final Evaluation ---
    scores = model.evaluate(X_test, y_test, verbose=0)

    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype("int32")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

    print("\n--- Bi-LSTM Model Performance ---")
    print("Accuracy= %.4f%%" % (accuracy * 100))
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    plot_cmat(y_test, y_pred)


if __name__ == '__main__':
    run_lstm_model()