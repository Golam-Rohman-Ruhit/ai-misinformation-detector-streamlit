#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection
The Keras version of neural network
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns  # Plotting-এর জন্য Seaborn ব্যবহার করা হয়েছে
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
# Keras/TensorFlow imports
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical


# keras.utils.np_utils.to_categorical এর বদলে সরাসরি keras.utils.to_categorical ব্যবহার করা হয়েছে


def plot_cmat(yte, ypred):
    '''Plotting confusion matrix using seaborn'''
    # scikitplot এর এরর এড়াতে seaborn ব্যবহার করা হচ্ছে
    cm = confusion_matrix(yte, ypred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake (0)', 'Real (1)'],
                yticklabels=['Fake (0)', 'Real (1)'])
    plt.title('Keras Neural Net Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


def baseline_model(input_dim):
    '''Neural network with 3 hidden layers'''
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation='relu', kernel_initializer='normal'))
    model.add(Dense(2, activation="softmax", kernel_initializer='normal'))

    # Gradient Descent (Using learning_rate parameter instead of deprecated lr)
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    # Configure the learning process of the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def run_keras_model():
    # ফাইল নাম (xtr/xte এর বদলে xtr_shuffled.npy ব্যবহার করা হচ্ছে)
    XTR_FILE = './xtr_shuffled.npy'
    XTE_FILE = './xte_shuffled.npy'
    YTR_FILE = './ytr_shuffled.npy'
    YTE_FILE = './yte_shuffled.npy'

    if not os.path.isfile(XTR_FILE):
        print("Error: The necessary Doc2Vec vector files (.npy) were not found.")
        print("Please ensure you have successfully run 'python getEmbeddings.py' first.")
        return

    # Load data
    xtr_full = np.load(XTR_FILE)
    xte_full = np.load(XTE_FILE)
    ytr_full = np.load(YTR_FILE)
    yte_full = np.load(YTE_FILE)

    # Keras-এর জন্য লেবেল এনকোডিং এবং One-Hot Encoding
    label_encoder = LabelEncoder()

    # Training labels (ytr_full)
    label_encoder.fit(ytr_full)
    encoded_ytr = to_categorical(label_encoder.transform(ytr_full))

    # Test labels (yte_full)
    label_encoder.fit(yte_full)
    encoded_yte = to_categorical(label_encoder.transform(yte_full))

    # ইনপুট ডাইমেনশন
    input_dim = xtr_full.shape[1]

    # Model Training
    model = baseline_model(input_dim)

    print("\nTraining Keras model for 20 epochs...")

    model.fit(xtr_full, encoded_ytr,
              epochs=20,
              batch_size=64,
              verbose=0)

    print("Model Trained!")

    # Model Evaluation
    score = model.evaluate(xte_full, encoded_yte, verbose=0)

    # Predict and calculate metrics
    probabs = model.predict(xte_full, verbose=0)
    y_pred = np.argmax(probabs, axis=1)  # Predicted class (0 or 1)
    y_true = yte_full  # True labels

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    print("\n--- Keras Neural Net Model Performance ---")
    print(f"Accuracy: {accuracy * 100:.4f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Draw the confusion matrix
    plot_cmat(y_true, y_pred)


if __name__ == '__main__':
    run_keras_model()