#!/usr/bin/env python
import sys
from os import path as pt
from os import makedirs
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.native import logger as l

import numpy as np
import gensim

from keras import backend as K
from keras.models import Sequential, load_model
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, TimeDistributed, Dropout

from keras_transformer import get_model, decode

class Transformer:

    _logger = None

    def __init__(self, data = None, model = None, LSTM_Depth = 3, sequence_length = 30):
        self._logger = l.getLogger()
        self._logger.debug("pinkySpeaker.lib.model.Transformer.__init__()")

        ## _dataset and _model are the two member variables of the class
        self._raw_data = data
        self._model = model
        self._dataset = None

        self._lyric_sequence_length = sequence_length

        self._startToken = "</START>"
        self._endToken = "endfile" ## TODO
        self._padToken = "</PAD>"

        if data:
            self._initArchitecture(data)
        elif model:
            self._model = self._loadNNModel(model)
        self._logger.info("Transformer model")
        return

    def _initArchitecture(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._initArchitecture()")

        vocab_size, max_title_length, all_titles_length, inp_sentences = self._initNNModel(raw_data)
        self._initDataset(raw_data, vocab_size, max_title_length, all_titles_length, inp_sentences)

        return

    def _initNNModel(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._initNNModel()")
        self._logger.info("Initialize NN Model")

        inp_sent, max_title_length, all_titles_length = self._constructSentences(raw_data)
        word_model = self._initWordModel(inp_sent)
        pretrained_weights = word_model.wv.vectors
        vocab_size, _ = pretrained_weights.shape
        ## Any new sub-model should be registered here
        ## The according function should be written to initialize it
        self._model = { 'word_model'  : None,
                        'lyric_model' : None
                      }

        ## The order matters because of word2idx usage, therefore manual initialization here
        self._model['word_model'] = word_model
        self._model['Transformer'] = self._initLyricModel(pretrained_weights)

        self._logger.info("Transformer Compiled successfully")
        return vocab_size, max_title_length, all_titles_length, inp_sent

    def _loadNNModel(self, modelpath):

        return { 'word_model'   :   gensim.models.Word2Vec.load(pt.join(modelpath, "word_model.h5")),
                 'lyric_model'  :   load_model(pt.join(modelpath, "lyric_model.h5"))
               }

    def _initDataset(self, raw_data, vocab_size, mx_t_l, all_t_l, inp_sent):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._initDataset()")

        lyric_set = self._constructTLSet(raw_data, vocab_size, mx_t_l, all_t_l)

        if len(lyric_set['encoder_input']) != len(lyric_set['output']) or len(lyric_set['decoder_input']) != len(lyric_set['output']):
            raise ValueError("Wrong lyric set dimensions!")

        self._dataset = { 'word_model'      : inp_sent,
                          'lyric_model'     : lyric_set 
                     }

        self._logger.info("Dataset constructed successfully")
        return   

    def _initWordModel(self, inp_sentences):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._initWordModel()")
        inp_sentences.append([self._padToken]) # Token that ensembles masking of training weights. Used to pad sequence length
        inp_sentences.append([self._startToken]) # Token that ensembles the start of a sequence
        wm = gensim.models.Word2Vec(inp_sentences, size = 300, min_count = 1, window = 4, iter = 200)
        self._logger.info("Word2Vec word model initialized")
        return wm

    def _initTitleModel(self, weights, LSTM_Depth):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._initTitleModel()")

        vocab_size, embedding_size = weights.shape
        tm = Sequential()
        tm.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[weights]))
        tm.add(Dropout(0.2))
        for _ in range(LSTM_Depth - 1):
            tm.add(LSTM(units=embedding_size, return_sequences=True))
            tm.add(Dropout(0.2))
        tm.add(LSTM(units=2 * embedding_size, return_sequences=False))
        tm.add(Dropout(0.2))
        tm.add(Dense(units=vocab_size))
        tm.add(Activation('softmax'))
        tm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        self._logger.info("Title model initialized")
        self._logger.info(tm.summary())
        return tm

    def _initLyricModel(self, weights):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._initLyricModel()")

        vocab_size, embedding_size = weights.shape

        lm = get_model(
            token_num = vocab_size,
            embed_dim = embedding_size,
            encoder_num = 2,
            decoder_num = 2,
            head_num = 2,
            hidden_dim = 128,
            attention_activation = 'relu',
            feed_forward_activation = 'relu',
            dropout_rate = 0.05,
            embed_weights = weights,
            embed_trainable = False
        )

        lm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        self._logger.info("Transformer model initialized")
        self._logger.info(lm.summary())
        return lm

    def _setClassWeight(self, vocab_size):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._setClassWeight()")
        clw = {}
        for i in range(vocab_size):
            clw[i] = 1
        clw[self.word2idx(self._padToken)] = 0
        clw[self.word2idx(self._startToken)] = 0
        return clw

    def _constructSentences(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._constructSentences()")
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
        self._logger.debug("pinkySpeaker.lib.model.Transformer._listToChunksList()")
        chunk_list = []
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i: i + n])
        return chunk_list

    def _constructTLSet(self, raw_data, vocab_size, max_title_length, all_titles_length):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._constructTLSet()")

        # lyric_set = {
        #              'input'            : np.zeros([2, len(raw_data), self._lyric_sequence_length], dtype = np.int32),
        #              'sample_weight'    : np.zeros([len(raw_data), self._lyric_sequence_length], dtype = np.int32),
        #              # lyric target will be the output of a softmax, i.e. a float and should be considered as such.
        #              'output'           : np.zeros([len(raw_data), self._lyric_sequence_length, 1], dtype = np.float64) 
        #              }   

        encoder_input = []
        decoder_input = []
        decoder_output = []
        sample_weight = []

        ## Iterate over each song. Keep index
        for song in raw_data:

            ## We need a fixed sequence length. The next function will grab a song and will return NN inputs and targets, sized _lyric_sequence_length
            ## If the song is bigger than this, multiple pairs of inputs/target will be returned.
            encoder_in, decoder_in, decoder_target, song_sample_weight = self._splitSongtoSentence(" ".join([" ".join(x) for x in ([song['title']] + song['lyrics'])]).split())

            ## For each input/target pair...
            for enc_in, dec_in, dec_out, weight in zip(encoder_in, decoder_in, decoder_target, song_sample_weight):
                ## Convert str array to embed index tensor
                ## And convert target str tokens to indices. Indices to one hot vecs vocab_size sized. Pass one-hot vecs through softmax to construct final target
                encoder_input.append(np.asarray([self.word2idx(x) for x in enc_in]))
                decoder_input.append(np.asarray([self.word2idx(x) for x in dec_in]))
                decoder_output.append(np.asarray([[self.word2idx(x)] for x in dec_out]))
                sample_weight.append(np.asarray(weight))

        lyric_set = {
                        'encoder_input'     : np.asarray(encoder_input, dtype = np.int32),
                        'decoder_input'     : np.asarray(decoder_input, dtype = np.int32),                        
                        'output'            : np.asarray(decoder_output, dtype = np.int32),
                        'sample_weight'     : np.asarray(sample_weight, dtype = np.int32)
                    }

        self._logger.info("Lyric encoder input tensor dimensions: {}".format(lyric_set['encoder_input'].shape))
        self._logger.info("Lyric decoder input tensor dimensions: {}".format(lyric_set['decoder_input'].shape))
        self._logger.info("Lyric Target tensor dimensions: {}".format(lyric_set['output'].shape))

        return lyric_set

    ## Receives an input tensor and returns an elem-by-elem softmax computed vector of the same dims
    def _softmax(self, inp_tensor):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._softmax()")
        m = np.max(inp_tensor)
        e = np.exp(inp_tensor - m)
        return e / np.sum(e)

    def _splitSongtoSentence(self, curr_song):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._splitSongtoSentence()")
        
        step = self._lyric_sequence_length - 1

        encoder_input  = [ [self._startToken] + curr_song[x : min(len(curr_song), x + step)] for x in range(0, len(curr_song), step)]
        decoder_input  = [ curr_song[x : min(len(curr_song), x + step)] + [self._endToken] for x in range(0, len(curr_song), step)]
        decoder_output = decoder_input

        ## Pad input and output sequence to match the batch sequence length
        encoder_input[-1]  += [self._padToken] * (step + 1 - len(encoder_input[-1]))
        decoder_input[-1]  += [self._padToken] * (step + 1 - len(decoder_input[-1]))
        decoder_output[-1] += [self._padToken] * (step + 1 - len(decoder_output[-1]))

        song_sample_weight = [[     0 if x == self._padToken
                               else 0 if x == self._startToken
                               else 50 if x == "endfile" 
                               else 10 if x == "endline" 
                               else 1 for x in inp] 
                            for inp in encoder_input]

        return encoder_input, decoder_input, decoder_output, song_sample_weight

    def _setClassWeight(self, vocab_size):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._setClassWeight()")
        clw = {}
        for i in range(vocab_size):
            clw[i] = 1
        clw[self.word2idx("endline")] = 50
        return clw

    ## Receive a word, return the index in the vocabulary
    def word2idx(self, word):
        self._logger.debug("pinkySpeaker.lib.model.Transformer.word2idx()")
        return self._model['word_model'].wv.vocab[word].index
    ## Receive a vocab index, return the workd
    def idx2word(self, idx):
        self._logger.debug("pinkySpeaker.lib.model.Transformer.idx2word()")
        return self._model['word_model'].wv.index2word[idx]
    ## Receive a vocab index, return its one hot vector
    def idx2onehot(self, idx, size):
        self._logger.debug("pinkySpeaker.lib.model.Transformer.idx2onehot()")
        ret = np.zeros(size)
        ret[idx] = 1000
        return ret

    ## Converts "endline" to '\n' for pretty printing
    ## Also masks meta-tokens but throws a warning
    def _prettyPrint(self, text):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._prettyPrint()")

        if self._startToken in text:
            self._logger.warning("</START> has been found to generated text!")
        if self._padToken in text:
            self._logger.warning("</PAD> has been found to generated text!")
        if "endline" in text:
            self._logger.warning("Endline found in text!")

        return text.replace("endline ", "\n").replace("endfile", "\nEND")

    ## Just fit it!
    def fit(self, epochs = 50, save_model = None):
        self._logger.debug("pinkySpeaker.lib.model.Transformer.fit()")

        # Build a small toy token dictionary
        tokens = 'all work and no play makes jack a dull boy'.split(' ')
        token_dict = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
        }
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)

        # Generate toy data
        encoder_inputs_no_padding = []
        encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
        for i in range(1, len(tokens) - 1):
            encode_tokens, decode_tokens = tokens[:i], tokens[i:]
            encode_tokens = ['<START>'] + encode_tokens + ['<END>'] + ['<PAD>'] * (len(tokens) - len(encode_tokens))
            output_tokens = decode_tokens + ['<END>', '<PAD>'] + ['<PAD>'] * (len(tokens) - len(decode_tokens))
            decode_tokens = ['<START>'] + decode_tokens + ['<END>'] + ['<PAD>'] * (len(tokens) - len(decode_tokens))
            encode_tokens = list(map(lambda x: token_dict[x], encode_tokens))
            decode_tokens = list(map(lambda x: token_dict[x], decode_tokens))
            output_tokens = list(map(lambda x: [token_dict[x]], output_tokens))
            encoder_inputs_no_padding.append(encode_tokens[:i + 2])
            encoder_inputs.append(encode_tokens)
            decoder_inputs.append(decode_tokens)
            decoder_outputs.append(output_tokens)

        ## TODO: You are here. Check input dimensions.
        ## Fork example to see how it works
        hist = self._model['Transformer'].fit( x = [self._dataset['lyric_model']['encoder_input'], self._dataset['lyric_model']['decoder_input']],
                                               y = self._dataset['lyric_model']['output'],
                                               # sample_weight = self._dataset['lyric_model']['sample_weight'],
                                               batch_size = 4,
                                               epochs = 50,
                                               callbacks = [LambdaCallback(on_epoch_end=self._lyrics_per_epoch)] )
       
        if save_model:
            save_model = pt.join(save_model, "Transformer")
            makedirs(save_model, exist_ok = True)
            self._model['Transformer'].save(pt.join(save_model, "Transformer.h5"))
        return hist.history['loss']

    ## Run a model prediction based on sample input
    def predict(self, seed, load_model = None):
        self._logger.debug("pinkySpeaker.lib.model.Transformer.predict()")

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
        lyrics = self._generate_next(title, self._model['Transformer'], False, num_generated = 540)
        lyrics = " ".join(lyrics.split()[len(title.split()):])

        self._logger.info("\nSeed: {}\nSong Title\n{}\nLyrics\n{}".format(seed, self._prettyPrint(title), self._prettyPrint(lyrics)))

        return

    ## Booting callback on title generation between epochs
    def _title_per_epoch(self, epoch, _):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._title_per_epoch()")

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
            _sample = self._generate_next(text, self._model['title_model'], title = True, num_generated = 20)
            self._logger.info('%s... -> \n%s\n' % (text, self._prettyPrint(_sample)))
        return

    ## Booting callback on lyric generation between epochs
    def _lyrics_per_epoch(self, epoch, _):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._lyrics_per_epoch()")

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
            _sample = self._generate_next(text, self._model['Transformer'], title = False)
            self._logger.info('\n%s... -> \n%s\n' % (text, self._prettyPrint(_sample)))
        return

    ## Model sampling setup function
    def _generate_next(self, text, model, title, num_generated = 320):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._generate_next()")

        word_idxs = [self.word2idx(word) for word in text.lower().split()]
        print(word_idxs)
        prediction = decode(
                            model,
                            word_idxs,
                            start_token = self.word2idx(self._startToken),
                            end_token = self.word2idx(self._endToken),
                            pad_token = self.word2idx(self._padToken),
                            max_len = num_generated,
                            top_k = 10,
                            temperature = 1.0
                    )

        # for i in range(num_generated):
        #     prediction = decode(
        #                         model,
        #                         np.array(word_idxs),
        #                         start_token = self.word2idx(self._startToken),
        #                         end_token = self.word2idx(self._endToken),
        #                         pad_token = self.word2idx(self._padToken),
        #                         max_len = num_generated,
        #                         top_k = 10,
        #                         temperature = 1.0
        #                 )
        #     prediction = model.predict(x=np.array(word_idxs))
        #     max_cl = 0
        #     max_indx = 0
        #     samples = prediction[-1] if title else prediction[-1][0]
        #     for ind, item in enumerate(samples):  ## TODO plz fix this for title model
        #         if item > max_cl:
        #             max_cl = item
        #             max_indx = ind

        #     idx = self._sample(samples, temperature=0.7)
        #     word_idxs.append(idx)

        #     if self.idx2word(idx) == "endfile" or (title and self.idx2word(idx) == "endline"):
        #         break

        return ' '.join(self.idx2word(idx) for idx in word_idxs + prediction)

    ## Take prediction vector, return the index of most likely class
    def _sample(self, preds, temperature=1.0):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._sample()")

        if temperature <= 0:
            return np.argmax(preds)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
