# import Keras library
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM, Input, Bidirectional, SpatialDropout1D
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import Recall, Precision
from keras import backend as K
from sklearn.metrics import classification_report


# other library
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each tweet
MAX_SEQUENCE_LENGTH = 10
# This is fixed.
EMBEDDING_DIM = 100


df = pd.read_csv('./Data/train_set.csv', index_col=[0])
# df = df.iloc[:300, :]
df = df.dropna()
# print(df['label'].value_counts().plot(kind='barh', color='r'))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['filtered_tweet'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['filtered_tweet'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
print(X[0])

Y = pd.get_dummies(df['label']).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.10, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def recall_at_k(y_pred, y_true, k):
    ct = 0
    for i in range(len(y_true)):
        if y_true[i] in y_pred[i][0:k]:
            ct += 1
    return ct / len(y_true)


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(
    LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(15, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc', f1_m, precision_m, recall_m])

epochs = 5
batch_size = 64
training_time = time.time()
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[
    EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
print('Training time is', time.time() - training_time, 's')

testing_time = time.time()
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, Y_test)

probs = model.predict(X_test)

# p = pd.DataFrame(probs)
# p['y_true'] = Y_test.argmax(axis=1)
# p.to_csv('lstm_probs.csv', index=False)


print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}\n  F1 Score: {:0.3f}\n  Precision: {:0.3f}\n  Recall: {:0.3f}'.format(
    loss, accuracy, f1_score, precision, recall))
print('Training time is', time.time() - testing_time, 's')


# # Evaluate TEST model class prediction accuracy
# print("[INFO] Evaluating network...")
# predictions = model.predict(X_test, batch_size=batch_size)
# target_names = ['0', '1', '2', '3', '4', '5', '7',
#                 '8', '11', '13', '14', '15', '16', '17', '19']
# print(classification_report(Y_test.argmax(axis=1),
#                             predictions.argmax(axis=1),
#                             target_names=target_names))
# preds = pd.DataFrame()
# preds['y_pred'] = predictions.argmax(axis=1)
# preds['y_true'] = Y_test.argmax(axis=1)
# preds.to_csv('bi_lstm_preds.csv', index=False)
