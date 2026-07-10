#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection
The naive bayes model
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import seaborn as sns


# getEmbeddings আমদানি করার আর প্রয়োজন নেই, কারণ ডেটা আগে থেকেই সেভ করা আছে।


def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    # scikitplot এর এরর এড়াতে seaborn ব্যবহার করা হচ্ছে
    cm = confusion_matrix(yte, ypred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake (0)', 'Real (1)'],
                yticklabels=['Fake (0)', 'Real (1)'])
    plt.title('Naive Bayes Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


def run_naive_bayes_model():
    # ফাইল নাম (getEmbeddings.py দ্বারা তৈরি)
    XTR_FILE = './xtr_shuffled.npy'
    XTE_FILE = './xte_shuffled.npy'
    YTR_FILE = './ytr_shuffled.npy'
    YTE_FILE = './yte_shuffled.npy'

    # ডেটা ফাইল তৈরির চেক
    if not os.path.isfile(XTR_FILE):
        print("Error: The necessary Doc2Vec vector files (.npy) were not found.")
        print("Please ensure you have successfully run 'python getEmbeddings.py' first.")
        return

    # ডেটা লোড করা হচ্ছে
    xtr = np.load(XTR_FILE)
    xte = np.load(XTE_FILE)
    ytr = np.load(YTR_FILE)
    yte = np.load(YTE_FILE)

    # Use the built-in Naive Bayes classifier
    print("Training Gaussian Naive Bayes model...")
    gnb = GaussianNB()
    gnb.fit(xtr, ytr)

    print("Predicting on test data...")
    y_pred = gnb.predict(xte)

    # Evaluation (SVM-এর মতো একই মেট্রিক্স ব্যবহার করা হচ্ছে)
    accuracy = accuracy_score(yte, y_pred)
    precision = precision_score(yte, y_pred, average='binary', zero_division=0)
    recall = recall_score(yte, y_pred, average='binary', zero_division=0)
    f1 = f1_score(yte, y_pred, average='binary', zero_division=0)

    print("\n--- Naive Bayes Model Performance ---")
    print(f"Accuracy: {accuracy * 100:.4f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Draw the confusion matrix
    plot_cmat(yte, y_pred)


# মূল ফাংশন কল করা হচ্ছে
if __name__ == '__main__':
    print("Running Naive Bayes Classification...")
    run_naive_bayes_model()