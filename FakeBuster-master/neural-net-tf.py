#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection
The TensorFlow version of neural network
(Rewritten using TensorFlow 2.x Custom Training Loop)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Hyperparameters
IN_DIM = 300
CLASS_NUM = 2
LEARN_RATE = 0.01
EPOCHS = 20
BATCH_SIZE = 64


def plot_cmat(yte, ypred):
    '''Plotting confusion matrix using seaborn'''
    cm = confusion_matrix(yte, ypred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake (0)', 'Real (1)'],
                yticklabels=['Fake (0)', 'Real (1)'])
    plt.title('TensorFlow Neural Net Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


# --- TensorFlow 2.x Model Class ---
class FakeNewsModel(tf.keras.Model):
    def __init__(self):
        super(FakeNewsModel, self).__init__()
        # Layer 1
        self.d1 = tf.keras.layers.Dense(300, activation='relu')
        self.do1 = tf.keras.layers.Dropout(0.4)
        # Layer 2
        self.d2 = tf.keras.layers.Dense(300, activation='relu')
        self.do2 = tf.keras.layers.Dropout(0.4)
        # Layer 3
        self.d3 = tf.keras.layers.Dense(300, activation='relu')
        self.do3 = tf.keras.layers.Dropout(0.4)
        # Output Layer
        self.out = tf.keras.layers.Dense(CLASS_NUM)

    def call(self, x, training=False):
        x = self.d1(x)
        if training:
            x = self.do1(x)
        x = self.d2(x)
        if training:
            x = self.do2(x)
        x = self.d3(x)
        if training:
            x = self.do3(x)
        return self.out(x)


def run_tf_model():
    XTR_FILE = './xtr_shuffled.npy'
    XTE_FILE = './xte_shuffled.npy'
    YTR_FILE = './ytr_shuffled.npy'
    YTE_FILE = './yte_shuffled.npy'

    # 1. ফাইল চেক
    if not os.path.isfile(XTR_FILE):
        print("Error: The necessary data files (.npy) were not found.")
        print("Please run 'python getEmbeddings.py' first.")
        return

    print("Loading data...")
    xtr = np.load(XTR_FILE, allow_pickle=True)
    xte = np.load(XTE_FILE, allow_pickle=True)
    ytr = np.load(YTR_FILE)
    yte = np.load(YTE_FILE)

    # 2. ডেটা টাইপ চেক
    if xtr.ndim < 2 or isinstance(xtr[0], str) or isinstance(xtr[0], np.str_):
        print("\n❌ Error: Data appears to be RAW TEXT strings (from LSTM run).")
        print("⚠️ ACTION REQUIRED: You MUST run 'python getEmbeddings.py' first to regenerate vectors.")
        return

    # ডেটা টাইপ কাস্টিং
    xtr = xtr.astype(np.float32)
    xte = xte.astype(np.float32)
    ytr = ytr.astype(np.int32)
    yte = yte.astype(np.int32)

    # 3. tf.data.Dataset তৈরি
    train_ds = tf.data.Dataset.from_tensor_slices((xtr, ytr)).shuffle(10000).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((xte, yte)).batch(BATCH_SIZE)

    # 4. মডেল, লস এবং অপটিমাইজার
    model = FakeNewsModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARN_RATE, momentum=0.9)

    # মেট্রিক্স
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # 5. কাস্টম ট্রেনিং লুপ
    print(f"\nStarting TensorFlow training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        # FIX: reset_states() -> reset_state()
        train_loss.reset_state()
        train_accuracy.reset_state()

        for images, labels in train_ds:
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)

        print(f'Epoch {epoch + 1}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result() * 100:.2f}%')

    # 6. ইভালুয়েশন
    print("\nEvaluating model on test data...")
    predictions_list = []
    true_labels_list = []

    for images, labels in test_ds:
        preds = model(images, training=False)
        preds_cls = tf.argmax(preds, axis=1)
        predictions_list.extend(preds_cls.numpy())
        true_labels_list.extend(labels.numpy())

    y_pred = np.array(predictions_list)
    y_true = np.array(true_labels_list)

    # মেট্রিক্স
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    print("\n--- TensorFlow Neural Net Model Performance ---")
    print(f"Accuracy: {accuracy * 100:.4f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    plot_cmat(y_true, y_pred)


if __name__ == "__main__":
    run_tf_model()