#!/usr/bin/env python
import sys
from os import path as pt
from os import makedirs
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.native import logger as l

import numpy as np
import gensim

from keras import backend as K
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, TimeDistributed, Flatten
from keras.models import Sequential, load_model
from keras.utils.data_utils import get_file

from keras.utils import to_categorical

class simpleRNN:

    _logger = None

    def __init__(self, data = None):
        self._logger = l.getLogger()
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.__init__()")

        ## _dataset and _model are the two member variables of the class
        self._raw_data = data
        self._dataset = None
        self._model = None

        self._lyric_sequence_length = 320

        if data:
            self._initArchitecture(data)
        self._logger.info("SimpleRNN model")
        return

    def _initArchitecture(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initArchitecture()")

        vocab_size, max_title_length, all_titles_length, inp_sentences = self._initNNModel(raw_data)
        self._initDataset(raw_data, vocab_size, max_title_length, all_titles_length, inp_sentences)

        return

    def _initNNModel(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initNNModel()")
        self._logger.info("Initialize NN Model")

        inp_sent, max_title_length, all_titles_length = self._constructSentences(raw_data)
        word_model = self._initWordModel(inp_sent)
        pretrained_weights = word_model.wv.vectors
        vocab_size, _ = pretrained_weights.shape
        ## Any new sub-model should be registered here
        ## The according function should be written to initialize it
        self._model = { 'word_model'  : word_model,
                        'title_model' : self._initTitleModel(pretrained_weights),
                        'lyric_model' : self._initLyricModel(pretrained_weights) 
                      }
        self._logger.info("SimpleRNN Compiled successfully")

        return vocab_size, max_title_length, all_titles_length, inp_sent

    def _initDataset(self, raw_data, vocab_size, mx_t_l, all_t_l, inp_sent):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initDataset()")

        title_set, lyric_set = self._constructTLSet(raw_data, vocab_size, mx_t_l, all_t_l)

        if len(title_set['input']) != len(title_set['output']):
            raise ValueError("Wrong title set dimensions!")
        if len(lyric_set['input']) != len(lyric_set['output']):
            raise ValueError("Wrong lyric set dimensions!")

        self._dataset = { 'word_model'      : inp_sent,
                          'title_model'     : title_set,
                          'lyric_model'     : lyric_set 
                     }

        self._logger.info("Dataset constructed successfully")
        return   

    def _initWordModel(self, inp_sentences):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initWordModel()")
        wm = gensim.models.Word2Vec(inp_sentences, size = 300, min_count = 1, window = 4, iter = 200)
        self._logger.info("Word2Vec word model initialized")
        return wm

    def _initTitleModel(self, weights):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initTitleModel()")

        vocab_size, embedding_size = weights.shape
        tm = Sequential()
        tm.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[weights]))
        tm.add(LSTM(units=2*embedding_size, return_sequences=True))
        tm.add(LSTM(units=2*embedding_size))
        tm.add(Dense(units=vocab_size))
        tm.add(Activation('softmax'))
        tm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        self._logger.info("Title model initialized")
        self._logger.info(tm.summary())
        return tm

    def _initLyricModel(self, weights):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initLyricModel()")

        vocab_size, embedding_size = weights.shape

        lm = Sequential()
        lm.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, trainable = False, weights=[weights]))
        lm.add(LSTM(units=2*embedding_size, input_shape = (None, embedding_size), return_sequences=True))
        lm.add(LSTM(units=2*embedding_size, input_shape = (None, 2*embedding_size), return_sequences = True))
        # tm.add(Flatten())
        lm.add(TimeDistributed(Dense(units=vocab_size, activation = 'softmax')))
        # tm.add(Flatten())
        # tm.add(Dense(units = vocab_size, activation = 'softmax'))
        # tm.add(Activation('softmax'))
        lm.compile(optimizer='adam', loss='categorical_crossentropy')
        self._logger.info("Lyric model initialized")
        self._logger.info(lm.summary())
        return lm

    def _listToChunksList(self, lst, n):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._listToChunksList()")
        chunk_list = []
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i: i + n])
        return chunk_list

    def _constructSentences(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._constructSentences()")
        self._logger.info("Sentence preprocessing for word model")

        sentence_size = 10
        max_title_length = 0
        all_titles_length = 0
        words = []
        for song in raw_data:

            curr_title_length = len(song['title'])
            all_titles_length += curr_title_length - 1
            if curr_title_length > max_title_length:
                max_title_length = curr_title_length
            
            for word in song['title']:
                words.append(word)
            for sent in song['lyrics']:
                for word in sent:
                    words.append(word)
        return self._listToChunksList(words, sentence_size), max_title_length, all_titles_length

    def _constructTLSet(self, raw_data, vocab_size, max_title_length, all_titles_length):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._constructTLSet()")

        title_set = {
                     'input': np.zeros([all_titles_length, max_title_length], dtype=np.int32), 
                     'output': np.zeros([all_titles_length], dtype=np.int32)
                     }

        lyric_set = {
                     'input': np.zeros([len(raw_data), self._lyric_sequence_length], dtype = np.int32),
                     # lyric target will be the output of a softmax, i.e. a float and should be considered as such.
                     'output': np.zeros([len(raw_data), self._lyric_sequence_length, vocab_size], dtype = np.float64) 
                     }   

        title_sample_index = 0
        ## Iterate over each song. Keep index
        for songIdx, song in enumerate(raw_data):

            ## Iterate over length of current song
            for curr_sent_size in range(len(song['title']) - 1):
                
                ## Traverse all indices until current index ( + 1 exists in the range to grab the next one as a target)
                for current_index in range(curr_sent_size + 1):
                    title_set['input'][title_sample_index][(max_title_length - 1) - curr_sent_size + current_index] = self.word2idx(song['title'][current_index])
                    title_set['output'][title_sample_index] = self.word2idx(song['title'][current_index + 1])
                title_sample_index += 1

            ## At this point, title_set has been constructed.

            ## We need a fixed sequence length. The next function will grab a song and will return NN inputs and targets, sized _lyric_sequence_length
            ## If the song is bigger than this, multiple pairs of inputs/target will be returned.
            l_in, l_out = self._splitSongtoSentence(" ".join([" ".join(x) for x in ([song['title']] + song['lyrics'])]).split())

            ## For each input/target pair...
            for inp, out in zip(l_in, l_out):
                ## Convert str array to embed index tensor
                lyric_set['input'][songIdx] = np.asarray([self.word2idx(x) for x in inp])
                ## And convert target str tokens to indices. Indices to one hot vecs vocab_size sized. Pass one-hot vecs through softmax to construct final target
                lyric_set['output'][songIdx] = self._softmax(np.asarray([self.idx2onehot(self.word2idx(x), vocab_size) for x in out]))


                # print(self._softmax(np.asarray([self.idx2onehot(self.word2idx(x), vocab_size) for x in out])))
                # print(self._softmax(np.asarray([self.idx2onehot(0, vocab_size) for x in out])))
                # lyric_inputs += np.asarray([self.word2idx(x) for x in inp])
                # lyric_expected_outputs += np.asarray([self.word2idx(x) for x in out])

            # lyric_inputs.append(np.asarray([self.word2idx(x) for x in l_in]))
            # lyric_expected_outputs.append(np.asarray([self.word2idx(x) for x in l_out]))

            # flat_song = " ".join([" ".join([self.word2idx(token) for token in line]) for line in song['lyrics']]).split()
            # ind_song = " ".join([ " ".join([self.word2idx(token) for token in line]) for line in song['lyrics']]).split()
            # ind_song = [self.word2idx(tok) for line in [song['title']] + song['lyrics'] for tok in line]

            # lyric_inputs.append(np.asarray(ind_song[:-1]))
            # lyric_expected_outputs.append( np.asarray([ self.idx2onehot(x, 2917) for x in ind_song[1:] ]))


            # for indx in range(len(flat_song) - 4):
            #     lyric_inputs.append([self.word2idx(x) for x in flat_song[indx : indx + 4]])
            #     lyric_expected_outputs.append(self.word2idx(flat_song[indx + 4]))


        print(lyric_set['input'].shape)
        print(lyric_set['output'].shape)

        return title_set, lyric_set

    def _softmax(self, inp_tensor):
        m = np.max(inp_tensor)
        e = np.exp(inp_tensor - m)
        return e / np.sum(e)

    def _splitSongtoSentence(self, song_list):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._splitSongtoSentence()")
        
        l_in = [song_list[x : min(len(song_list), x + self._lyric_sequence_length)] for x in range(0, len(song_list), self._lyric_sequence_length)]
        l_out = [song_list[x + 1 : min(len(song_list), x + 1 + self._lyric_sequence_length)] for x in range(0, len(song_list), self._lyric_sequence_length)]
        l_in[-1] += ['endfile'] * (self._lyric_sequence_length - len(l_in[-1]))
        l_out[-1] += ['endfile'] * (self._lyric_sequence_length - len(l_out[-1]))

        # Works fine!
        return l_in, l_out

    def _dtLyricGenerator(self):

        while True:
            sequence_length = np.random.randint(10, 100)
            x_train = np.random.random((1000, sequence_length, 5))
            # y_train will depend on past 5 timesteps of x
            y_train = x_train[:, :, 0]
            for i in range(1, 5):
                y_train[:, i:] += x_train[:, :-i, i]
            y_train = to_categorical(y_train > 2.5)
            yield x_train, y_train

    def word2idx(self, word):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.word2idx()")
        return self._model['word_model'].wv.vocab[word].index
    def idx2word(self, idx):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.idx2word()")
        return self._model['word_model'].wv.index2word[idx]
    def idx2onehot(self, idx, size):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.idx2onehot()")
        ret = np.zeros(size)
        ret[idx] = 1000
        return ret

    def fit(self, save_model = None):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.fit()")

        # title_hist = self._model['title_model'].fit(self._dataset['title_model']['input'], 
        #                                             self._dataset['title_model']['output'],
        #                                             batch_size = 128,
        #                                             epochs = 2,
        #                                             callbacks = [LambdaCallback(on_epoch_end=self._title_per_epoch)] )

        # lyric_hist = self._model['lyric_model'].fit(test_in,
        #                                             test_out,
        #                                             batch_size = 4,
        #                                             epochs = 2,
        #                                             callbacks = [LambdaCallback(on_epoch_end=self._lyrics_per_epoch)] )


        # print("START")
        # test_indx = 80
        # print(self._dataset['lyric_model']['input'][test_indx].shape)
        # print(self._dataset['lyric_model']['output'][test_indx].shape)
        # # print(self._dataset['lyric_model']['input'][test_indx])
        # print(" ".join([self.idx2word(int(x)) for x in self._dataset['lyric_model']['input'][test_indx]]))
        # # print(self._dataset['lyric_model']['output'][test_indx])
        # temp_list = []
        # for token_vec in self._dataset['lyric_model']['output'][test_indx]:
        #     print(token_vec.shape)
        #     temp_list.append(self.idx2word(self._sample(token_vec)))
        # print(" ".join(temp_list))

        # print(" ".join([self.idx2word(int(x)) for x in self._dataset['lyric_model']['input'][80]]))
        # print(" ".join([self.idx2word(np.argmax(x)) for x in self._dataset['lyric_model']['output'][80]]))

        # for id in range(126):
        #     print("YAY")
        #     x = [self.idx2word(int(x)) for x in self._dataset['lyric_model']['input'][id]]
        #     y = [self.idx2word(np.argmax(x)) for x in self._dataset['lyric_model']['output'][id]]
        #     for i in range(319):
        #         assert(x[i + 1] == y[i])

        # exit(1)



        lyric_hist = self._model['lyric_model'].fit(self._dataset['lyric_model']['input'],
                                                    self._dataset['lyric_model']['output'],
                                                    batch_size = 4,
                                                    epochs = 30,
                                                    callbacks = [LambdaCallback(on_epoch_end=self._lyrics_per_epoch)] )
       
        if save_model:
            save_model = pt.join(save_model, "simpleRNN")
            makedirs(save_model, exist_ok = True)
            self._model['word_model'].save(pt.join(save_model, "word_model.h5"))
            self._model['title_model'].save(pt.join(save_model, "title_model.h5"))
            self._model['lyric_model'].save(pt.join(save_model, "lyric_model.h5"))
        return

    def _generateSynthetic(self):

        num_songs = 126
        max_song_len = 55
        synthetic_in, synthetic_out = [], []
        songs = []

        # print(len(self._raw_data))
        for song in self._raw_data:
            songindx = [self.word2idx(x) for x in song['title']]
            for line in song['lyrics']:
                songindx += [self.word2idx(x) for x in line]
            # max_song_len = max(max_song_len, len(songindx))
            songs.append(songindx + [0] * (max_song_len - len(songindx)))

        ## OK. Now songs contains 126 entries (1 per song, where each entry has the max len of song)
        synthetic_in = np.asarray(songs)
        synthetic_out = np.asarray(songs)
        return synthetic_in, synthetic_out

    def _title_per_epoch(self, epoch, _):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._title_per_epoch()")

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
            _sample = self._generate_next(text, self._model['title_model'], title = True)
            print('%s... -> %s' % (text, _sample))
        return

    def _lyrics_per_epoch(self, epoch, _):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._lyrics_per_epoch()")

        print('\nGenerating text after epoch: %d' % epoch)
        texts = [
                'dark side',
                'another brick in the wall',
                'echoes',
                'high hopes',
                'shine on you crazy diamond',
                'breathe',
                'have a cigar',
                'comfortably numb'
            ]
        for text in texts:
            sample = self._generate_next(text, self._model['lyric_model'], title = False)
            print('%s... -> %s' % (text, sample))
        return

    def _generate_next(self, text, model, title, num_generated=320):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._generate_next()")

        word_idxs = [self.word2idx(word) for word in text.lower().split()]
        init_endline_bias = model.layers[-2].weights[1][self.word2idx("endline")]
        init_endfile_bias = model.layers[-2].weights[1][self.word2idx("endfile")]
        # print(text)
        for i in range(num_generated):
            prediction = model.predict(x=np.array(word_idxs))
            # print(prediction.shape)
            # print(prediction[-1][0].shape)
            max_cl = 0
            max_indx = 0
            for ind, item in enumerate(prediction[-1][0]):
                if item > max_cl:
                    max_cl = item
                    max_indx = ind


            # print(self.idx2word(max_indx))
            # print(max_indx)
            idx = self._sample(prediction[-1][0], temperature=0.7)
            if idx == 0:
                print("WARNING!!!\n\n\n\n\n\n\n\n\n\n")
            if self.idx2word(idx) == "endfile":
                print("EDNFILLLELELELELLE \n\n\n\n\n\n\n")
            word_idxs.append(idx)
            if (title == True and (self.idx2word(idx) == "endline" or self.idx2word(idx) == "endfile")) or (title == False and self.idx2word(idx) == "endfile"):
                break
            else:
            	pass
                # if self.idx2word(idx) == "endline":
                #     K.set_value(model.layers[-2].weights[1][self.word2idx("endline")], init_endline_bias)

                # if title == True:
                #     b = 2
                # else:
                #     b = 0.2
        #         K.set_value(model.layers[-2].weights[1][self.word2idx("endline")], model.layers[-2].weights[1][self.word2idx("endline")] + b*abs(model.layers[-2].weights[1][self.word2idx("endline")]))
        #         K.set_value(model.layers[-2].weights[1][self.word2idx("endfile")], model.layers[-2].weights[1][self.word2idx("endfile")] + 0.4*abs(model.layers[-2].weights[1][self.word2idx("endfile")]))
        # K.set_value(model.layers[-2].weights[1][self.word2idx("endline")], init_endline_bias)
        # K.set_value(model.layers[-2].weights[1][self.word2idx("endfile")], init_endfile_bias)

        return ' '.join(self.idx2word(idx) for idx in word_idxs)


    def _sample(self, preds, temperature=1.0):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._sample()")

        if temperature <= 0:
            return np.argmax(preds)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)