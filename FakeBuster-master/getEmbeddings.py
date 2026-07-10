#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection
The Doc2Vec pre-processing and Text Cleaning
"""

import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from nltk.corpus import stopwords
import nltk
import os

# NLTK stopwords ডাউনলোড নিশ্চিত করা
try:
    stopwords.words("english")
except LookupError:
    print("NLTK 'stopwords' resource not found. Downloading...")
    nltk.download('stopwords')


def textClean(text):
    """
    Get rid of the non-letter and non-number characters
    """
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)


def cleanup(text):
    if not isinstance(text, str):
        return ""
    text = textClean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def constructLabeledSentences(data):
    sentences = []
    for index, row in data.items():
        sentences.append(TaggedDocument(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences


def getEmbeddings(path, vector_dimension=300):
    """
    Generate Doc2Vec training and testing data (Vectors)
    """
    data = pd.read_csv(path)
    data.dropna(subset=['text', 'label'], inplace=True)
    data = data.reset_index(drop=True).drop(['id'], axis=1, errors='ignore')

    for i in range(len(data)):
        data.loc[i, 'text'] = cleanup(data.loc[i, 'text'])

    x = constructLabeledSentences(data['text'])
    y = data['label'].values

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7,
                         epochs=10, seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.epochs)

    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        train_labels[i] = y[i]

    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        test_labels[j] = y[i]
        j = j + 1

    return text_train_arrays, text_test_arrays, train_labels, test_labels


def clean_data():
    """
    Generate processed string and save as .npy files (For LSTM)
    """
    print("Executing clean_data()... Processing raw text from CSV.")
    path = 'datasets/train.csv'

    # ডেটা লোড এবং প্রসেসিং
    data = pd.read_csv(path)
    data.dropna(subset=['text', 'label'], inplace=True)
    data = data.drop(['id'], axis=1, errors='ignore')
    data = data.reset_index(drop=True)

    for i in range(len(data)):
        data.loc[i, 'text'] = cleanup(data.loc[i, 'text'])

    # শাফলিং
    data = data.sample(frac=1).reset_index(drop=True)

    x = data.loc[:, 'text'].values
    y = data.loc[:, 'label'].values

    train_size = int(0.8 * len(y))

    xtr = x[:train_size]
    xte = x[train_size:]
    ytr = y[:train_size]
    yte = y[train_size:]

    # টেক্সট স্ট্রিং হিসেবে সেভ করা হচ্ছে
    np.save('xtr_shuffled.npy', xtr)
    np.save('xte_shuffled.npy', xte)
    np.save('ytr_shuffled.npy', ytr)
    np.save('yte_shuffled.npy', yte)
    print("Raw text data saved successfully.")


if __name__ == '__main__':
    # Default behavior: Create Doc2Vec vectors for SVM/NaiveBayes/Keras
    if os.path.isfile('datasets/train.csv'):
        print("Generating Doc2Vec embeddings and saving training data...")
        xtr_vec, xte_vec, ytr, yte = getEmbeddings("datasets/train.csv")
        np.save('xtr_shuffled.npy', xtr_vec)
        np.save('xte_shuffled.npy', xte_vec)
        np.save('ytr_shuffled.npy', ytr)
        np.save('yte_shuffled.npy', yte)
        print("Doc2Vec Embeddings saved successfully.")
    else:
        print("Error: 'datasets/train.csv' not found.")