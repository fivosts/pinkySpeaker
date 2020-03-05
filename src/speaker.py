#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import gensim
import string
import os

from keras import backend as K
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.utils.data_utils import get_file
import re

DATASET_PATH="../dataset/"

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

# for i in sentences[0:50]:
# 	print(i)
# 	print("\n\n\n")
word2vecmodel = None
title = True

def fetch_data():

	dataset = []
	for file in os.listdir(DATASET_PATH):
		with open(DATASET_PATH +file, 'r') as f:
			song = []
			for line in f:
				if line != "\n":
					sentence = re.sub(".*?\[(.*?)\]", "", line).lower().replace("i'm", "i am")\
																		.replace("it's", "it is")\
																		.replace("isn't", "is not")\
																		.replace("there's", "there is")\
																		.replace("they've", "they have")\
																		.replace("\n", " endline")\
																		.replace("we've", "we have")\
																		.replace("wasn't", "was not")\
																		.replace(".", " . ")\
																		.replace(",", " , ")\
																		.replace("-", "")\
																		.replace("\"", "")\
																		.replace(":", "")\
																		.replace("(", "")\
																		.replace(")", "")\
																		.replace("?", " ?")\
																		.replace("!", " !")\
																		.split()
					if sentence:
						song.append(sentence)
			artist = "_".join(song[0][:-1])
			title = "_".join(song[1][:-1])
			song[-1].append("endfile")
			dataset.append({'artist': song[0], 'title': song[1], 'lyrics': song[2:]})

	return dataset

def struct_sentences(dataset):

	sentences = []
	max_len = 0
	for song in dataset:
		s = [song['title']] + song['lyrics']
		for sen in s:
			if len(sen) > max_len:
				max_len = len(sen)
		sentences += s
	# print(sentences)
	return sentences, max_len

def set_title_trainset(dataset, word_model):
	title_length = 0
	set_length = 0
	for song in dataset:
		set_length += len(song['title']) - 1
		if len(song['title']) > title_length:
			title_length = len(song['title'])

	print(set_length, title_length)
	tset = {'input': np.zeros([set_length, title_length], dtype=np.int32), 'output': np.zeros([set_length], dtype=np.int32)}

	index = 0
	for j, song in enumerate(dataset):
		## i will go from 0->15
		## i means: I will add this number of words. 1 word up to len()-1
		for i in range(len(song['title']) - 1):
			# print("i: {}".format(i))
			## k will go from 0->i+1
			## k means: if i am adding i words, start counting each one with k
			for k in range(i + 1):
				# print("k: {}".format(k))
				# print(song['title'][k])
				# print(song['title'][k+1])
				# print(title_length - 1 + (k - i))

				max_index = title_length - 1
				sentence_size = i
				current = k
				# print("Going to insert {} in position: {}".format(song['title'][k], max_index - sentence_size + current))

				tset['input'][index][max_index - sentence_size + current] = word2idx(song['title'][k], word_model)
				tset['output'][index] = word2idx(song['title'][k+1], word_model)
			index += 1

	# print("Input array: {}\nOutput array: {}".format(tset['input'], tset['output']))
	# for i in range(len(tset['input'])):
	# 	print("Input array: {}\nOutput array: {}".format(tset['input'][i], tset['output'][i]))

	return tset

def set_lyric_trainset(dataset, word_model):

	inputs = []
	outputs = []

	for song in dataset:
		flat_song = [song['title']] + song['lyrics']
		flat_song = [" ".join(x) for x in flat_song]
		flat_song = " ".join(flat_song).split()
		for i in range(len(flat_song) - 4):
			inputs.append([word2idx(x, word_model) for x in flat_song[i : i + 4]])
			outputs.append(word2idx(flat_song[i + 4], word_model))

	lset = {'input': np.zeros([len(inputs), 4], dtype=np.int32), 'output': np.zeros([len(inputs)], dtype=np.int32)}

	lset['input'] = np.asarray(inputs, dtype = np.int32)
	lset['output'] = np.asarray(outputs, dtype = np.int32)


	return lset

def trainWordModel(inp):

	word_model = gensim.models.Word2Vec(inp, size=300, min_count=1, window=4, iter=200)
	pretrained_weights = word_model.wv.vectors
	vocab_size, embedding_size = pretrained_weights.shape
	print(vocab_size)
	print(embedding_size)
	print(pretrained_weights.shape)
	print('Result embedding shape:', pretrained_weights.shape)
	print('Checking similar words:')
	for word in ['dark', 'side', 'of', 'the', 'moon', 'endline', 'endfile']:
		most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.wv.most_similar(word)[:8])
		print('  %s -> %s' % (word, most_similar))

	word_model.save("word2vec.model")

	# token = 'another'
	# counter = 0
	# visited = set()
	# visited.add(token)
	# sampled_title = [token]
	# print("  {} -> {}".format(token, ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.wv.most_similar(token)[:8])))
	# while (token != 'endline' or token != 'endfile') and counter < 10:
	# 	token = word_model.wv.most_similar(token)[0][0]

	# 	i = 0
	# 	nextToken = word_model.wv.most_similar(token)[i][0]
	# 	while nextToken in visited and i < len(word_model.wv.most_similar(nextToken)) - 1:
	# 		i += 1
	# 		nextToken = word_model.wv.most_similar(token)[i][0]
	# 	token = nextToken
	# 	visited.add(token)

	# 	counter += 1
	# 	print("  {} -> {}".format(token, ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.wv.most_similar(token)[:8])))
	# 	sampled_title.append(token)
	# print(" ".join(sampled_title))

	return word_model

def word2idx(word, word_model):
  return word_model.wv.vocab[word].index
def idx2word(idx, word_model):
  return word_model.wv.index2word[idx]






def sample(preds, temperature=1.0):
	if temperature <= 0:
		return np.argmax(preds)
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

def generate_next(text, num_generated=140):
	word_idxs = [word2idx(word, word2vecmodel) for word in text.lower().split()]
	# print(model.layers[-2])
	# print(model.layers[-2].weights[1])
	init_endline_bias = model.layers[-2].weights[1][word2idx("endline", word2vecmodel)]
	init_endfile_bias = model.layers[-2].weights[1][word2idx("endfile", word2vecmodel)]
	for i in range(num_generated):
	# while True:
		prediction = model.predict(x=np.array(word_idxs))

		idx = sample(prediction[-1], temperature=0.7)
		word_idxs.append(idx)
		# if idx2word(idx, word2vecmodel) == "endline" or idx2word(idx, word2vecmodel) == "endfile":
		if (title == True and (idx2word(idx, word2vecmodel) == "endline" or idx2word(idx, word2vecmodel) == "endfile")) or (title == False and idx2word(idx, word2vecmodel) == "endfile"):
			break
		else:
			if idx2word(idx, word2vecmodel) == "endline":
				K.set_value(model.layers[-2].weights[1][word2idx("endline", word2vecmodel)], init_endline_bias)

			if title == True:
				b = 2
			else:
				b = 0.2
			K.set_value(model.layers[-2].weights[1][word2idx("endline", word2vecmodel)], model.layers[-2].weights[1][word2idx("endline", word2vecmodel)] + b*abs(model.layers[-2].weights[1][word2idx("endline", word2vecmodel)]))
			K.set_value(model.layers[-2].weights[1][word2idx("endfile", word2vecmodel)], model.layers[-2].weights[1][word2idx("endfile", word2vecmodel)] + 0.4*abs(model.layers[-2].weights[1][word2idx("endfile", word2vecmodel)]))
	K.set_value(model.layers[-2].weights[1][word2idx("endline", word2vecmodel)], init_endline_bias)
	K.set_value(model.layers[-2].weights[1][word2idx("endfile", word2vecmodel)], init_endfile_bias)

	return ' '.join(idx2word(idx, word2vecmodel) for idx in word_idxs)

def on_epoch_end(epoch, _):
	print('\nGenerating text after epoch: %d' % epoch)
	texts = [
			'dark',
			'dark side',
			'another',
			'echoes',
			'high',
			'shine',
			'on',
			'have',
			'comfortably'
		]
	for text in texts:
		sample = generate_next(text)
		print('%s... -> %s' % (text, sample))
	return

model = None

def main():
	data = fetch_data()
	sentences, max_sentence_len = struct_sentences(data)

	word_model = trainWordModel(sentences)
	global word2vecmodel
	word2vecmodel = word_model
	print(word2vecmodel)
	title_set = set_title_trainset(data, word_model)
	lyric_set = set_lyric_trainset(data, word_model)

	assert len(title_set['input']) == len(title_set['output']), "Wrong title set dimensions"
	assert len(lyric_set['input']) == len(lyric_set['output']), "Wrong lyric set dimensions"

	for i in range(len(title_set['input'])):
		print("  {}  ->  {}".format("".join(str(title_set['input'][i])), title_set['output'][i]))
		print(len(title_set['input']))

	print(max_sentence_len)

	print('\nPreparing the data for LSTM...')
	# train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
	# train_y = np.zeros([len(sentences)], dtype=np.int32)
	# for i, sentence in enumerate(sentences):
	# 	for t, word in enumerate(sentence[:-1]):
	# 		train_x[i, t] = word2idx(word)
	# 		train_y[i] = word2idx(sentence[-1])
	# print("Input-> output:")
	# print(sentence[:-1])
	# print(sentence[-1])
	## Input is a sentence without the last word
	## Target is the last word of the sentence

	## (7200, 40)
	# print('train_x shape:', train_x.shape)
	# print('train_y shape:', train_y.shape)

	print('\nTraining title LSTM...')

	pretrained_weights = word_model.wv.vectors
	vocab_size, embedding_size = pretrained_weights.shape

	print("Vocab size: {}, embedding size: {}".format(vocab_size, embedding_size))
	print("Size of title examples: {}".format(len(title_set['input'])))

	global title
	title = True

	title_model = Sequential()
	title_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights]))
	title_model.add(LSTM(units=2*embedding_size, return_sequences=True))
	title_model.add(LSTM(units=2*embedding_size))
	title_model.add(Dense(units=vocab_size))
	title_model.add(Activation('softmax'))
	title_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
	global model
	model = title_model
	hist = title_model.fit(title_set['input'], title_set['output'],
	          batch_size=4,
	          epochs=30,
	          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

	title = False
	lyric_model = Sequential()
	lyric_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights]))
	lyric_model.add(LSTM(units=2*embedding_size, return_sequences=True))
	lyric_model.add(LSTM(units=2*embedding_size))
	lyric_model.add(Dense(units=vocab_size))
	lyric_model.add(Activation('softmax'))
	lyric_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

	model = lyric_model
	l_hist = lyric_model.fit(lyric_set['input'], lyric_set['output'],
	          batch_size=16,
	          epochs=40,
	          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

	lyric_model.save("lyric_model.h5")
	title_model.save("title_model.h5")

	return

def predict():

	print("PREDICT")
	title_model = load_model("title_model.h5")
	lyric_model = load_model("lyric_model.h5")

	global word2vecmodel
	word2vecmodel = gensim.models.Word2Vec.load("word2vec.model")

	inp = input()
	global model
	global title
	title = True
	model = title_model

	T = generate_next(inp)

	model = lyric_model
	title = False
	L = " ".join(generate_next(T).split()[len(T.split()):])

	print("TITLE")
	print(T.replace("endline", "\n"))
	print(L.replace("endline ", "\n").replace("endfile", ""))


if __name__ == "__main__":
	main()
	# predict()
	exit(0)