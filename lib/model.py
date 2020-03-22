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
from keras.layers import Dense, Activation, TimeDistributed, Dropout
from keras.models import Sequential, load_model
from keras.utils.data_utils import get_file

from keras.utils import to_categorical

class simpleRNN:

    _logger = None

    def __init__(self, data = None, model = None, LSTM_Depth = 3):
        self._logger = l.getLogger()
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.__init__()")

        ## _dataset and _model are the two member variables of the class
        self._raw_data = data
        self._model = model
        self._dataset = None

        self._lyric_sequence_length = 320
        self._maskToken = "MASK_TOKEN"
        self._startToken = "START_TOKEN"

        if data:
            self._initArchitecture(data, LSTM_Depth)
        elif model:
            self._model = self._loadNNModel(model)
        self._logger.info("SimpleRNN model")
        return

    def _initArchitecture(self, raw_data, LSTM_Depth):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initArchitecture()")

        vocab_size, max_title_length, all_titles_length, inp_sentences = self._initNNModel(raw_data, LSTM_Depth)
        self._initDataset(raw_data, vocab_size, max_title_length, all_titles_length, inp_sentences)

        return

    def _initNNModel(self, raw_data, LSTM_Depth):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initNNModel()")
        self._logger.info("Initialize NN Model")

        inp_sent, max_title_length, all_titles_length = self._constructSentences(raw_data)
        word_model = self._initWordModel(inp_sent)
        pretrained_weights = word_model.wv.vectors
        vocab_size, _ = pretrained_weights.shape
        ## Any new sub-model should be registered here
        ## The according function should be written to initialize it
        self._model = { 'word_model'  : None,
                        'title_model' : None,
                        'lyric_model' : None
                      }

        ## The order matters because of word2idx usage, therefore manual initialization here
        self._model['word_model'] = word_model
        self._model['title_model'] = self._initTitleModel(pretrained_weights, LSTM_Depth)
        self._model['lyric_model'] = self._initLyricModel(pretrained_weights, LSTM_Depth)

        self._logger.info("SimpleRNN Compiled successfully")
        return vocab_size, max_title_length, all_titles_length, inp_sent

    def _loadNNModel(self, modelpath):

        return { 'word_model'   :   gensim.models.Word2Vec.load(pt.join(modelpath, "word_model.h5")),
                 'title_model'  :   load_model(pt.join(modelpath, "title_model.h5")),
                 'lyric_model'  :   load_model(pt.join(modelpath, "lyric_model.h5"))
               }

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
        inp_sentences.append([self._maskToken]) # Token that ensembles masking of training weights. Used to pad sequence length
        inp_sentences.append([self._startToken]) # Token that ensembles the start of a sequence
        wm = gensim.models.Word2Vec(inp_sentences, size = 300, min_count = 1, window = 4, iter = 200)
        self._logger.info("Word2Vec word model initialized")
        return wm

    def _initTitleModel(self, weights, LSTM_Depth):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initTitleModel()")

        vocab_size, embedding_size = weights.shape
        tm = Sequential()
        tm.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[weights]))
        tm.add(Dropout(0.2))
        for _ in range(LSTM_Depth):
	        tm.add(LSTM(units=embedding_size, return_sequences=True))
    	    tm.add(Dropout(0.2))
        tm.add(Dense(units=vocab_size))
        tm.add(Activation('softmax'))
        tm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        self._logger.info("Title model initialized")
        self._logger.info(tm.summary())
        return tm

    def _initLyricModel(self, weights, LSTM_Depth):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initLyricModel()")

        vocab_size, embedding_size = weights.shape

        lm = Sequential()
        # lm.add(Masking(mask_value = self.word2idx("endline"), input_shape = ([None])))
        lm.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, trainable = False, weights=[weights]))
        lm.add(Dropout(0.2))
        for _ in range(LSTM_Depth):
	        lm.add(LSTM(units=embedding_size, input_shape = (None, embedding_size), return_sequences=True))
    	    lm.add(Dropout(0.2))
        lm.add(TimeDistributed(Dense(units=vocab_size, activation = 'softmax')))
        lm.add(TimeDistributed(Dropout(0.2)))
        lm.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode = "temporal")

        self._logger.info("Lyric model initialized")
        self._logger.info(lm.summary())
        return lm

    def _setClassWeight(self, vocab_size):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._setClassWeight()")
        clw = {}
        for i in range(vocab_size):
            clw[i] = 1
        clw[self.word2idx(self._maskToken)] = 0
        clw[self.word2idx(self._startToken)] = 0
        return clw

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

    def _listToChunksList(self, lst, n):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._listToChunksList()")
        chunk_list = []
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i: i + n])
        return chunk_list

    def _constructTLSet(self, raw_data, vocab_size, max_title_length, all_titles_length):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._constructTLSet()")

        title_set = {
                     'input'            : np.zeros([all_titles_length, max_title_length], dtype=np.int32), 
                     'output'           : np.zeros([all_titles_length], dtype=np.int32)
                     }

        lyric_set = {
                     'input'            : np.zeros([len(raw_data), self._lyric_sequence_length], dtype = np.int32),
                     'sample_weight'    : np.zeros([len(raw_data), self._lyric_sequence_length], dtype = np.int32),
                     # lyric target will be the output of a softmax, i.e. a float and should be considered as such.
                     'output'           : np.zeros([len(raw_data), self._lyric_sequence_length, vocab_size], dtype = np.float64) 
                     }   

        title_sample_index = 0
        ## Iterate over each song. Keep index
        for songIdx, song in enumerate(raw_data):

            song_title = song['title']
            ## Iterate over length of current song
            for curr_sent_size in range(len(song_title) - 1):
                
                ## Traverse all indices until current index ( + 1 exists in the range to grab the next one as a target)
                for current_index in range(curr_sent_size + 1):
                    title_set['input'][title_sample_index][(max_title_length - 1) - curr_sent_size + current_index] = self.word2idx(song_title[current_index])
                    title_set['output'][title_sample_index] = self.word2idx(song_title[current_index + 1])
                title_sample_index += 1

            ## At this point, title_set has been constructed.

            ## We need a fixed sequence length. The next function will grab a song and will return NN inputs and targets, sized _lyric_sequence_length
            ## If the song is bigger than this, multiple pairs of inputs/target will be returned.
            song_spl_inp, song_spl_out, song_sample_weight = self._splitSongtoSentence(" ".join([" ".join(x) for x in ([song_title] + song['lyrics'])]).split())

            ## For each input/target pair...
            for inp, out, weight in zip(song_spl_inp, song_spl_out, song_sample_weight):
                ## Convert str array to embed index tensor
                lyric_set['input'][songIdx] = np.asarray([self.word2idx(x) for x in inp])
                lyric_set['sample_weight'][songIdx] = np.asarray(weight)
                ## And convert target str tokens to indices. Indices to one hot vecs vocab_size sized. Pass one-hot vecs through softmax to construct final target
                lyric_set['output'][songIdx] = self._softmax(np.asarray([self.idx2onehot(self.word2idx(x), vocab_size) for x in out]))

        self._logger.info("Title Input tensor dimensions: {}".format(title_set['input'].shape))
        self._logger.info("Title Target tensor dimensions: {}".format(title_set['output'].shape))

        self._logger.info("Lyric Input tensor dimensions: {}".format(lyric_set['input'].shape))
        self._logger.info("Lyric Target tensor dimensions: {}".format(lyric_set['output'].shape))

        return title_set, lyric_set

    ## Receives an input tensor and returns an elem-by-elem softmax computed vector of the same dims
    def _softmax(self, inp_tensor):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._softmax()")
        m = np.max(inp_tensor)
        e = np.exp(inp_tensor - m)
        return e / np.sum(e)

    def _splitSongtoSentence(self, song_list):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._splitSongtoSentence()")
        
        song_list.insert(0, self._startToken)

        song_spl_inp = [song_list[x : min(len(song_list), x + self._lyric_sequence_length)] for x in range(0, len(song_list), self._lyric_sequence_length)]
        song_spl_out = [song_list[x + 1 : min(len(song_list), x + 1 + self._lyric_sequence_length)] for x in range(0, len(song_list), self._lyric_sequence_length)]

        ## Pad input and output sequence to match the batch sequence length
        song_spl_inp[-1] += [self._maskToken] * (self._lyric_sequence_length - len(song_spl_inp[-1]))
        song_spl_out[-1] += [self._maskToken] * (self._lyric_sequence_length - len(song_spl_out[-1]))

        song_sample_weight = [[0 if x == self._maskToken else 50 if x == "endfile" else 50 if x == "endline" else 1 for x in inp] for inp in song_spl_inp]

        return song_spl_inp, song_spl_out, song_sample_weight

    ## Receive a word, return the index in the vocabulary
    def word2idx(self, word):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.word2idx()")
        return self._model['word_model'].wv.vocab[word].index
    ## Receive a vocab index, return the workd
    def idx2word(self, idx):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.idx2word()")
        return self._model['word_model'].wv.index2word[idx]
    ## Receive a vocab index, return its one hot vector
    def idx2onehot(self, idx, size):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.idx2onehot()")
        ret = np.zeros(size)
        ret[idx] = 1000
        return ret

    ## Converts "endline" to '\n' for pretty printing
    ## Also masks meta-tokens but throws a warning
    def _prettyPrint(self, text):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._prettyPrint()")

        if self._startToken in text:
            self._logger.warning("START_TOKEN has been found to generated text!")
        elif self._maskToken in text:
            self._logger.warning("MASK_TOKEN has been found to generated text!")

        return text.replace("endline ", "\n").replace("endfile", "\nEND")

    ## Just fit it!
    def fit(self, save_model = None):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.fit()")

        title_hist = self._model['title_model'].fit(self._dataset['title_model']['input'], 
                                                    self._dataset['title_model']['output'],
                                                    batch_size = 128,
                                                    epochs = 2,
                                                    callbacks = [LambdaCallback(on_epoch_end=self._title_per_epoch)] )

        lyric_hist = self._model['lyric_model'].fit(self._dataset['lyric_model']['input'],
                                                    self._dataset['lyric_model']['output'],
                                                    batch_size = 8,
                                                    epochs = 60,
                                                    sample_weight = self._dataset['lyric_model']['sample_weight'],
                                                    callbacks = [LambdaCallback(on_epoch_end=self._lyrics_per_epoch)] )
       
        if save_model:
            save_model = pt.join(save_model, "simpleRNN")
            makedirs(save_model, exist_ok = True)
            self._model['word_model'].save(pt.join(save_model, "word_model.h5"))
            self._model['title_model'].save(pt.join(save_model, "title_model.h5"))
            self._model['lyric_model'].save(pt.join(save_model, "lyric_model.h5"))
        return

    ## Run a model prediction based on sample input
    def predict(self, seed, load_model = None):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.predict()")

        if not self._model and not load_model:
            self._logger.critical("Load model path has not been provided! Predict failed!")
            raise ValueError("Model is not cached. Model load path has not been provided. Predict failed!")
        else:
            if load_model:
                if self._model:
                    ## TODO not necessarily
                    self._logger.info("New model has been provided. Overriding cached model...")
                self._model = self._loadNNModel(load_model)

        title = self._generate_next(seed, self._model['title_model'], True, num_generated = 10)
        lyrics = self._generate_next(title, self._model['lyric_model'], False, num_generated = 540)
        lyrics = " ".join(lyrics.split()[len(title.split()):])

        self._logger.info("\nSeed: {}\nSong Title\n{}\nLyrics\n{}".format(seed, self._prettyPrint(title), self._prettyPrint(lyrics)))

        return

    ## Booting callback on title generation between epochs
    def _title_per_epoch(self, epoch, _):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._title_per_epoch()")

        self._logger.info('\nGenerating text after epoch: %d' % epoch)
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
            self._logger.info('%s... -> \n%s\n' % (text, self._prettyPrint(_sample)))
        return

    ## Booting callback on lyric generation between epochs
    def _lyrics_per_epoch(self, epoch, _):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._lyrics_per_epoch()")

        self._logger.info('\nGenerating text after epoch: %d' % epoch)
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
            _sample = self._generate_next(text, self._model['lyric_model'], title = False)
            self._logger.info('\n%s... -> \n%s\n' % (text, self._prettyPrint(_sample)))
        return

    ## Model sampling setup function
    def _generate_next(self, text, model, title, num_generated = 320):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._generate_next()")

        word_idxs = [self.word2idx(word) for word in text.lower().split()]
        # init_endline_bias = model.layers[-2].weights[1][self.word2idx("endline")]
        # init_endfile_bias = model.layers[-2].weights[1][self.word2idx("endfile")]
        for i in range(num_generated):
            prediction = model.predict(x=np.array(word_idxs))
            max_cl = 0
            max_indx = 0
            samples = prediction[-1] if title else prediction[-1][0]
            for ind, item in enumerate(samples):  ## TODO plz fix this for title model
                if item > max_cl:
                    max_cl = item
                    max_indx = ind

            idx = self._sample(samples, temperature=0.7)
            word_idxs.append(idx)

            if self.idx2word(idx) == "endfile" or (title and self.idx2word(idx) == "endline"):
                break

        return ' '.join(self.idx2word(idx) for idx in word_idxs)

    ## Take prediction vector, return the index of most likely class
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