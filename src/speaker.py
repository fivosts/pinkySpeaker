#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import gensim
import string
import os

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file
import re

DATASET_PATH="/home/fivosts/PhD/Code/pinkySpeaker/dataset/"

# print('\nFetching the text...')
# url = 'https://raw.githubusercontent.com/maxim5/stanford-tensorflow-tutorials/master/data/arxiv_abstracts.txt'
# path = get_file('arxiv_abstracts.txt', origin=url)

# print('\nPreparing the sentences...')
# max_sentence_len = 40
# with open(path) as file_:
#   docs = file_.readlines()
# sentences = [[word for word in doc.lower().split()[:max_sentence_len]] for doc in docs]
# print(type(sentences))
# print(len(sentences))

def fetch_data():

	dataset = []
	for file in os.listdir(DATASET_PATH):
		with open(DATASET_PATH +file, 'r') as f:
			song = []
			for line in f:
				if line != "\n":
					result = re.sub(".*?\[(.*?)\]", "", line).lower().replace("\n", " endline").replace(".", "").replace(",", "").replace("-", "").replace("(", "").replace(")", "").replace("?", "").split()
					if result:
						song.append(result)
			artist = "_".join(song[0][:-1])
			title = "_".join(song[1][:-1])
			dataset.append({'artist': song[0], 'title': song[1], 'lyrics': song[2:]})

	return dataset

fetch_data()

def trainWordModel(inp):

	word_model = gensim.models.Word2Vec(inp, size=100, min_count=1, window=5, iter=1)
	pretrained_weights = word_model.wv.vectors
	vocab_size, emdedding_size = pretrained_weights.shape
	print(vocab_size)
	print(emdedding_size)
	print(pretrained_weights.shape)
	print('Result embedding shape:', pretrained_weights.shape)
	print('Checking similar words:')
	for word in ['model', 'network', 'train', 'learn']:
		most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.wv.most_similar(word)[:8])
		print('  %s -> %s' % (word, most_similar))

	return



# def word2idx(word):
#   return word_model.wv.vocab[word].index
# def idx2word(idx):
#   return word_model.wv.index2word[idx]

# print('\nPreparing the data for LSTM...')
# train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
# train_y = np.zeros([len(sentences)], dtype=np.int32)
# for i, sentence in enumerate(sentences):
#   for t, word in enumerate(sentence[:-1]):
#     train_x[i, t] = word2idx(word)
#   train_y[i] = word2idx(sentence[-1])
#   # print("Input-> output:")
#   # print(sentence[:-1])
#   # print(sentence[-1])
#   ## Input is a sentence without the last word
#   ## Target is the last word of the sentence
# exit(1)
# ## (7200, 40)
# print('train_x shape:', train_x.shape)
# print('train_y shape:', train_y.shape)

# print('\nTraining LSTM...')
# model = Sequential()
# model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
# model.add(LSTM(units=emdedding_size))
# model.add(Dense(units=vocab_size))
# model.add(Activation('softmax'))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# def sample(preds, temperature=1.0):
#   if temperature <= 0:
#     return np.argmax(preds)
#   preds = np.asarray(preds).astype('float64')
#   preds = np.log(preds) / temperature
#   exp_preds = np.exp(preds)
#   preds = exp_preds / np.sum(exp_preds)
#   probas = np.random.multinomial(1, preds, 1)
#   return np.argmax(probas)

# def generate_next(text, num_generated=10):
#   word_idxs = [word2idx(word) for word in text.lower().split()]
#   for i in range(num_generated):
#     prediction = model.predict(x=np.array(word_idxs))
#     idx = sample(prediction[-1], temperature=0.7)
#     word_idxs.append(idx)
#   return ' '.join(idx2word(idx) for idx in word_idxs)

# def on_epoch_end(epoch, _):
#   print('\nGenerating text after epoch: %d' % epoch)
#   texts = [
#     'deep convolutional',
#     'simple and effective',
#     'a nonconvex',
#     'a',
#   ]
#   for text in texts:
#     sample = generate_next(text)
#     print('%s... -> %s' % (text, sample))

# model.fit(train_x, train_y,
#           batch_size=128,
#           epochs=20,
#           callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])