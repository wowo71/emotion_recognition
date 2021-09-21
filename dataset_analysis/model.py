# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import TextVectorization
# import numpy as np
# import pandas as pd
# from tensorflow.keras.layers import Embedding
# from tensorflow.keras import layers
#
# # from keras.preprocessing.text import Tokenizer
# #
# path_to_glove_file = "/home/wojtek/Desktop/glove.6B.100d.txt"
#
# embeddings_index = {}
# with open(path_to_glove_file) as f:
#     for line in f:
#         word, coefs = line.split(maxsplit=1)
#         coefs = np.fromstring(coefs, "f", sep=" ")
#         embeddings_index[word] = coefs
#         # print(embeddings_index)
#         # break
#
# print("Found %s word vectors." % len(embeddings_index))
# df = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/dataset/internet_dataset/final_dataset_v8_top_3000_spellchecked.csv", sep=',', error_bad_lines=False)
# # comment = df['comment']
# # label = df['emotion']
# # top_words = len(embeddings_index)
# # tokenizer = Tokenizer(num_words=top_words)
# # tokenizer.fit_on_texts(comment)
#
# vectorizer = TextVectorization()
# text_ds = df['comment']
# vectorizer.adapt(text_ds)
# print(vectorizer.get_vocabulary()[:5])
# output = vectorizer([["the cat sat on the mat"]])
# print(output.numpy()[0, :6])
# voc = vectorizer.get_vocabulary()
# word_index = dict(zip(voc, range(len(voc))))
# test = ["the", "cat", "sat", "on", "the", "mat"]
# print([word_index[w] for w in test])
# num_tokens = len(voc) + 2
# embedding_dim = 100
# hits = 0
# misses = 0
#
# # Prepare embedding matrix
# embedding_matrix = np.zeros((num_tokens, embedding_dim))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # Words not found in embedding index will be all-zeros.
#         # This includes the representation for "padding" and "OOV"
#         embedding_matrix[i] = embedding_vector
#         hits += 1
#     else:
#         misses += 1
# print("Converted %d words (%d misses)" % (hits, misses))
#
#
#
# embedding_layer = Embedding(
#     num_tokens,
#     embedding_dim,
#     embeddings_initializer=keras.initializers.Constant(embedding_matrix),
#     trainable=False,
# )
#
#
# int_sequences_input = keras.Input(shape=(None,), dtype="int64")
# embedded_sequences = embedding_layer(int_sequences_input)
# x = layers.Conv1D(128, 5, activation="relu", padding="same")(embedded_sequences)
# x = layers.MaxPooling1D(5)(x)
# x = layers.Conv1D(128, 5, activation="relu", padding="same")(x)
# x = layers.MaxPooling1D(5)(x)
# x = layers.Conv1D(128, 5, activation="relu", padding="same")(x)
# x = layers.GlobalMaxPooling1D()(x)
# x = layers.Dense(128, activation="relu")(x)
# x = layers.Dropout(0.5)(x)
# preds = layers.Dense(len(df['emotion'].unique()), activation="softmax")(x)
# model = keras.Model(int_sequences_input, preds)
# model.summary()
#
# print(df.dtypes)
# validation_split = 0.2
# num_validation_samples = int(validation_split * len(df))
# train_samples = df['comment'][:-num_validation_samples]
# val_samples = df['comment'][-num_validation_samples:]
# train_labels = df['emotion'][:-num_validation_samples]
# val_labels = df['emotion'][-num_validation_samples:]
#
# x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
# x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()
#
# y_train = np.array(train_labels)
# y_val = np.array(val_labels)
#
# model.compile(
#     loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
# )
# model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_val, y_val))

import numpy as np
import pandas as pd
# import matplotlib.pylab as plt
# from livelossplot import PlotLossesKeras
# np.random.seed(7)
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, LSTM
# from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.preprocessing import sequence
# from gensim.models import Word2Vec, KeyedVectors, word2vec
# import gensim
# from gensim.utils import simple_preprocess
# from keras.utils import to_categorical
# import pickle
# import h5py
# from time import time

df = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/dataset/internet_dataset/final_dataset_v8_top_3000_spellchecked.csv", sep=',', error_bad_lines=False)
X = df['emotion']
y = df['comment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
print("X_train shape: " + str(X_train.shape))
print("X_test shape: " + str(X_test.shape))
print("X_val shape: " + str(X_val.shape))
print("y_train shape: " + str(y_train.shape))
print("y_test shape: " + str(y_test.shape))
print("y_val shape: " + str(y_val.shape))


path_to_glove_file = "/home/wojtek/Desktop/glove.6B.100d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
        # print(embeddings_index)
        # break

mxlen = 50
nb_classes = 6
top_words = len(embeddings_index)
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

word_index = tokenizer.word_index
X_train = sequence.pad_sequences(sequences_train, maxlen=mxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=mxlen)
X_val = sequence.pad_sequences(sequences_val, maxlen=mxlen)

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
y_val = np_utils.to_categorical(y_val, nb_classes)

# validation_split = 0.2
# num_validation_samples = int(validation_split * len(df))
# train_samples = df['comment'][:-num_validation_samples]
# val_samples = df['comment'][-num_validation_samples:]
# train_labels = df['emotion'][:-num_validation_samples]
# val_labels = df['emotion'][-num_validation_samples:]
#
# x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
# x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()
#
# y_train = np.array(train_labels)
# y_val = np.array(val_labels)
#