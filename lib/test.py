from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.callbacks import LambdaCallback
# prepare sequence

## i.e. samples
num_songs = 126

## i.e. num of timesteps 
max_len_songs = 25

## i.e. features
vocab_size = 2917

embedding_size = 300

X = np.zeros((num_songs, max_len_songs))
y = np.ones((num_songs, max_len_songs, vocab_size))

print(X.shape)
print(y.shape)

# print(X1)
# define LSTM configuration
n_batch = 16
n_epoch = 20
# create LSTM
model = Sequential()
model.add(Embedding(input_dim = vocab_size, trainable = False, output_dim = embedding_size))
model.add(LSTM(2*embedding_size, input_shape=(max_len_songs, embedding_size), return_sequences=True))
model.add(LSTM(2*embedding_size, input_shape=(max_len_songs, 2*embedding_size), return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation = 'softmax')))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:,0]:
	print('%.1f' % value)



