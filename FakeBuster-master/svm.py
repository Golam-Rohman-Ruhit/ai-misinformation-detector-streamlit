#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection
The SVM model
"""

import numpy as np
from sklearn.svm import LinearSVC  # LinearSVC is generally faster
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import seaborn as sns


# getEmbeddings এখন ব্যবহার করা হচ্ছে না, কারণ ডেটা আগে থেকেই সেভ করা আছে।

def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    cm = confusion_matrix(yte, ypred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake (0)', 'Real (1)'],
                yticklabels=['Fake (0)', 'Real (1)'])
    plt.title('SVM Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


def run_svm_model():
    # ফাইল নাম
    XTR_FILE = './xtr_shuffled.npy'
    XTE_FILE = './xte_shuffled.npy'
    YTR_FILE = './ytr_shuffled.npy'
    YTE_FILE = './yte_shuffled.npy'

    # নিশ্চিত করা হচ্ছে যে ভেক্টর ফাইলগুলি বিদ্যমান
    if not os.path.isfile(XTR_FILE):
        print("Error: The necessary Doc2Vec vector files (.npy) were not found.")
        print("Please ensure you have successfully run 'python getEmbeddings.py' first to create them.")
        return

    # ডেটা লোড করা হচ্ছে
    xtr = np.load(XTR_FILE)
    xte = np.load(XTE_FILE)
    ytr = np.load(YTR_FILE)
    yte = np.load(YTE_FILE)

    # নিশ্চিত করা হচ্ছে যে লোড করা ডেটা সংখ্যাসূচক (Doc2Vec vectors)
    if xtr.ndim < 2 or xtr.dtype == object:
        print("Error: Loaded data is not numerical vector data. Check getEmbeddings.py and run it again.")
        return

    # Model Training
    print("Training SVM model (LinearSVC)...")
    clf = LinearSVC(max_iter=10000)
    clf.fit(xtr, ytr)

    print("Predicting on test data...")
    y_pred = clf.predict(xte)

    # Evaluation
    accuracy = accuracy_score(yte, y_pred)
    precision = precision_score(yte, y_pred, average='binary', zero_division=0)
    recall = recall_score(yte, y_pred, average='binary', zero_division=0)
    f1 = f1_score(yte, y_pred, average='binary', zero_division=0)

    print("\n--- SVM Model Performance ---")
    print(f"Accuracy: {accuracy * 100:.4f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Draw the confusion matrix
    plot_cmat(yte, y_pred)


if __name__ == '__main__':
    run_svm_model()