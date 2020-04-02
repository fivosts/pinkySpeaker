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

import tensorflow as tf
import tensorflow_datasets as tfds
from lib import utils

class TfTransformer:

    def __init__(self, data = None, model = None, batch_size = 4, LSTM_Depth = 3, sequence_length = 30):
        self._logger = l.getLogger()
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer.__init__()")

        ## _dataset and _model are the two member variables of the class
        self._raw_data = data
        self._model = model
        self._dataset = None

        self._lyric_sequence_length = sequence_length

        self._startToken = "</START>"
        self._endToken = "</END>" ## TODO
        self._padToken = "</PAD>"

        if data:
            self._initArchitecture(data, batch_size)
        elif model:
            self._model = self._loadNNModel(model)
        self._logger.info("TfTransformer model")
        return

    ## Booting function of NN Model + dataset initialization
    def _initArchitecture(self, raw_data, batch_size):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._initArchitecture()")

        self._initDataset(raw_data, batch_size)
        vocab_size, max_title_length, all_titles_length, inp_sentences = self._initNNModel(raw_data)

        return

    ## Booting function of NN Model initialization
    def _initNNModel(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._initNNModel()")
        self._logger.info("Initialize NN Model")

        ## Temp specs. Those will be added as parameters
        num_layers = 2
        d_model = 64
        dff = 64
        num_heads = 2

        input_vocab_size = self._model['tokenizer'].vocab_size + 2
        target_vocab_size = self._model['tokenizer'].vocab_size + 2
        dropout_rate = 0.1
        ##

        self._model['transformer'] = self._setupTransformer(num_layers,
                                                            d_model, 
                                                            dff, 
                                                            num_heads, 
                                                            input_vocab_size, 
                                                            target_vocab_size, 
                                                            dropout_rate
                                                            )
        self._model['optimizer'] = self._setupOptimizer(d_model,
                                                        adam_params
                                                        )

        self._logger.info("TfTransformer Assembled successfully")
        return

    ## Core function that assembles the transformer
    def _setupTransformer(self, num_layers, d_model, dff, num_heads, input_vocab_size, target_vocab_size, dropout_rate):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._setupTransformer()")
        return _Transformer(num_layers, 
                            d_model, 
                            num_heads, 
                            dff,
                            input_vocab_size, 
                            target_vocab_size, 
                            pe_input=input_vocab_size, 
                            pe_target=target_vocab_size,
                            rate=dropout_rate
                           )

    ## Core function that assembles the optimizer and training schedule
    def _setupOptimizer(self, d_model, adam_params = {'beta_1': 0.9, 'beta_2': 0.98, 'epsilon': 1e-9}):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._setupOptimizer()")

        learning_rate = _CustomSchedule(d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate,
                                            beta_1 = adam_params['beta_1'],
                                            beta_2 = adam_params['beta_2'],
                                            epsilon = adam_params['epsilon']
                                            )

        ## TODO call plotter for learning rate
        ## TODO learn what all these are
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        return {'adam': optimizer, 
                'loss_obj': loss_object,
                'loss': train_loss,
                'accuracy': train_accuracy
                }

    ## Loads a model from file.
    def _loadNNModel(self, modelpath):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._loadNNModel()")
        ## TODO
        return { 'word_model'   :   gensim.models.Word2Vec.load(pt.join(modelpath, "word_model.h5")),
                 'lyric_model'  :   load_model(pt.join(modelpath, "lyric_model.h5"))
               }

    ## Booting function of dataset creation.
    ## Assigns the dataset  to self._dataset
    def _initDataset(self, raw_data, batch_size):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._initDataset()")

        ##  TODO  Encoder class will do that 5. Fix positional encoding.
        ##      And that too: Optionally fix mesh plot

        intermediate_set = self._raw2TfString(raw_data)
        self._logger.info("String dataset constructed")
        if not self._model:
            self._model = {'tokenizer': None}
        self._model['tokenizer'] = self._initTokenizer(intermediate_set)
        self._dataset = self._preprocessDataset(intermediate_set, batch_size)

        self._logger.info("Dataset has been encoded successfully")
        return

    ## Converts song from a dict-list format to single string
    def _songList2songStr(self, song, delim = " "):
        return delim.join([delim.join(line) for line in [song['title']] + song['lyrics']])

    ## Convert raw_data in list format to tf.Dataset format of strings
    ## Arrange inputs with targets
    def _raw2TfString(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._raw2TfString()")

        ## str_dataset will be a MapDataset.
        ## Each of its entries will be a pair tuple in an input->target relationship
        ## Each element of the tuple is a string tensor.
        for en, song in enumerate(raw_data):
            datapoint = tf.data.Dataset.from_tensor_slices([self._songList2songStr(song)])
            datapoint = datapoint.map(lambda key: (key, key))
            if not en:
                str_dataset = datapoint
            else:
                str_dataset = str_dataset.concatenate(datapoint)
        return str_dataset

    ## Receive tf.Dataset as input
    ## Initialize tokenizer, construct vocabulary, return tokenizer
    ## Input Dataset is considered to be the return type of _raw2TfString
    def _initTokenizer(self, str_dataset, target_vocab_size = 2**13):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._initTokenizer()")

        tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                    (x.numpy() for x, _ in str_dataset),
                    target_vocab_size = target_vocab_size,
                    reserved_tokens = ["<ENDLINE>"]
        )
        self._tokSanityCheck(tokenizer, "This is a comfortably numb tokenizer ?!")
        return tokenizer

    ## Take tokenizer and a sample string
    ## Encode, decode and check decoded string is still the same as the original
    def _tokSanityCheck(self, tokenizer, sample_string):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._tokSanityCheck()")

        tokenized_string = tokenizer.encode(sample_string)
        original_string = tokenizer.decode(tokenized_string)
        self._logger.info('Tokenized string is {}'.format(tokenized_string))
        self._logger.info('The original string: {}'.format(original_string))

        for ts in tokenized_string:
            self._logger.info('{}\t---->\t{}'.format(ts, tokenizer.decode([ts])))

        assert original_string == sample_string
        return

    ## Take tf.Dataset in string format, encodes it and returns
    ## Returned set should be batched, shuffled, cached and ready to be fed to the Transformer
    def _preprocessDataset(self, str_dataset, batch_size, buffer_size = 20000):

        preprocessed_dataset = (
                    str_dataset
                    .map(self._pair_encode)
                    .cache()
                    .shuffle(buffer_size)  
        )
        dataset = (
                    preprocessed_dataset
                    .padded_batch(batch_size, padded_shapes = ([None], [None]))
                    .prefetch(tf.data.experimental.AUTOTUNE)
        )
        return dataset

    ## Boot function of (input, target) encoding
    def _pair_encode(self, inp, tar):
        res_inp, res_tar = tf.py_function(self._encode, [inp, tar], [tf.int64, tf.int64])
        res_inp.set_shape([None])
        res_tar.set_shape([None])
        return res_inp, res_tar

    ## Nested encoding function
    def _encode(self, inp, tar):
        inp = [self._model['tokenizer'].vocab_size] + self._model['tokenizer'].encode(inp.numpy()) + [self._model['tokenizer'].vocab_size+1]
        ## Target will not have the start token.
        ## We want output to be shifted one position to the right wrt the input
        tar = self._model['tokenizer'].encode(tar.numpy()) + [self._model['tokenizer'].vocab_size+1]
        return inp, tar

    ## Initialize and return word model
    def _initWordModel(self, inp_sentences):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._initWordModel()")
        inp_sentences.append([self._padToken]) # Token that ensembles masking of training weights. Used to pad sequence length
        inp_sentences.append([self._startToken]) # Token that ensembles the start of a sequence
        wm = gensim.models.Word2Vec(inp_sentences, size = 300, min_count = 1, window = 4, iter = 200)
        self._logger.info("Word2Vec word model initialized")
        return wm

    ## Function to initialize and return title model
    ## Needs to be fixed
    def _initTitleModel(self, weights, LSTM_Depth):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._initTitleModel()")

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

    ## Function to initialize and return lyric model
    def _initLyricModel(self, weights):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._initLyricModel()")

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

        self._logger.info("TfTransformer model initialized")
        self._logger.info(lm.summary())
        return lm

    ## Set class weights for pad token and start token to 0.
    def _setClassWeight(self, vocab_size):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._setClassWeight()")
        clw = {}
        for i in range(vocab_size):
            clw[i] = 1
        clw[self.word2idx(self._padToken)] = 0
        clw[self.word2idx(self._startToken)] = 0
        return clw

    ## Precompute raw data information to construct word model
    ## Returns a list of chunks of sentences, the max length of titles
    ## and the total length of titles
    def _constructSentences(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._constructSentences()")
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

    ## Converts a sentence-list of words-list to a list of chunks of sentences
    def _listToChunksList(self, lst, n):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._listToChunksList()")
        chunk_list = []
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i: i + n])
        return chunk_list

    def _constructTLSet(self, raw_data, vocab_size, max_title_length, all_titles_length):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._constructTLSet()")

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
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._softmax()")
        m = np.max(inp_tensor)
        e = np.exp(inp_tensor - m)
        return e / np.sum(e)

    def _splitSongtoSentence(self, curr_song):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._splitSongtoSentence()")
        
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
                               else 50 if x == self._endToken 
                               else 10 if x == "<ENDLINE>" 
                               else 1 for x in inp] 
                            for inp in encoder_input]

        return encoder_input, decoder_input, decoder_output, song_sample_weight

    def _setClassWeight(self, vocab_size):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._setClassWeight()")
        clw = {}
        for i in range(vocab_size):
            clw[i] = 1
        clw[self.word2idx("<ENDLINE>")] = 50
        return clw

    ## Receive a word, return the index in the vocabulary
    def word2idx(self, word):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer.word2idx()")
        return self._model['word_model'].wv.vocab[word].index
    ## Receive a vocab index, return the workd
    def idx2word(self, idx):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer.idx2word()")
        return self._model['word_model'].wv.index2word[idx]
    ## Receive a vocab index, return its one hot vector
    def idx2onehot(self, idx, size):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer.idx2onehot()")
        ret = np.zeros(size)
        ret[idx] = 1000
        return ret

    ## Converts "<ENDLINE>" to '\n' for pretty printing
    ## Also masks meta-tokens but throws a warning
    def _prettyPrint(self, text):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._prettyPrint()")

        if self._startToken in text:
            self._logger.warning("</START> has been found to generated text!")
        if self._padToken in text:
            self._logger.warning("</PAD> has been found to generated text!")
        if "<ENDLINE>" in text:
            self._logger.warning("Endline found in text!")

        return text.replace("<ENDLINE> ", "\n")

    ## Just fit it!
    def fit(self, epochs = 50, save_model = None):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer.fit()")

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
        hist = self._model['TfTransformer'].fit( x = [self._dataset['lyric_model']['encoder_input'], self._dataset['lyric_model']['decoder_input']],
                                               y = self._dataset['lyric_model']['output'],
                                               # sample_weight = self._dataset['lyric_model']['sample_weight'],
                                               batch_size = 4,
                                               epochs = 50,
                                               callbacks = [LambdaCallback(on_epoch_end=self._lyrics_per_epoch)] )
       
        if save_model:
            save_model = pt.join(save_model, "TfTransformer")
            makedirs(save_model, exist_ok = True)
            self._model['TfTransformer'].save(pt.join(save_model, "TfTransformer.h5"))
        return hist.history['loss']

    ## Run a model prediction based on sample input
    def predict(self, seed, load_model = None):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer.predict()")

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
        lyrics = self._generate_next(title, self._model['TfTransformer'], False, num_generated = 540)
        lyrics = " ".join(lyrics.split()[len(title.split()):])

        self._logger.info("\nSeed: {}\nSong Title\n{}\nLyrics\n{}".format(seed, self._prettyPrint(title), self._prettyPrint(lyrics)))

        return

    ## Booting callback on title generation between epochs
    def _title_per_epoch(self, epoch, _):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._title_per_epoch()")

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
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._lyrics_per_epoch()")

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
            _sample = self._generate_next(text, self._model['TfTransformer'], title = False)
            self._logger.info('\n%s... -> \n%s\n' % (text, self._prettyPrint(_sample)))
        return

    ## Model sampling setup function
    def _generate_next(self, text, model, title, num_generated = 320):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._generate_next()")

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

        #     if self.idx2word(idx) == "endfile" or (title and self.idx2word(idx) == "<ENDLINE>"):
        #         break

        return ' '.join(self.idx2word(idx) for idx in word_idxs + prediction)

    ## Take prediction vector, return the index of most likely class
    def _sample(self, preds, temperature=1.0):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._sample()")

        if temperature <= 0:
            return np.argmax(preds)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

class _MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(_MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
                
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)    # (batch_size, seq_len, d_model)
        k = self.wk(k)    # (batch_size, seq_len, d_model)
        v = self.wv(v)    # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)    # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)    # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)    # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
                q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])    # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                                                    (batch_size, -1, self.d_model))    # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)    # (batch_size, seq_len_q, d_model)
                
        return output, attention_weights

class _EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(_EncoderLayer, self).__init__()

        self.mha = _MultiHeadAttention(d_model, num_heads)
        self.ffn = utils.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)    # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)    # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.ffn(out1)    # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)    # (batch_size, input_seq_len, d_model)
        
        return out2


class _DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(_DecoderLayer, self).__init__()

        self.mha1 = _MultiHeadAttention(d_model, num_heads)
        self.mha2 = _MultiHeadAttention(d_model, num_heads)

        self.ffn = utils.point_wise_feed_forward_network(d_model, dff)
 
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
        
    def call(self, x, enc_output, training, 
                     look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)    # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(
                enc_output, enc_output, out1, padding_mask)    # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)    # (batch_size, target_seq_len, d_model)
        
        ffn_output = self.ffn(out2)    # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)    # (batch_size, target_seq_len, d_model)
        
        return out3, attn_weights_block1, attn_weights_block2     

class _Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                             maximum_position_encoding, rate=0.1):
        super(_Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = utils.positional_encoding(maximum_position_encoding, self.d_model)
        
        
        self.enc_layers = [_EncoderLayer(d_model, num_heads, dff, rate) 
                                             for _ in range(num_layers)]
    
        self.dropout = tf.keras.layers.Dropout(rate)
                
    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]
        
        # adding embedding and position encoding.
        x = self.embedding(x)    # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x    # (batch_size, input_seq_len, d_model)     

class _Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                             maximum_position_encoding, rate=0.1):
        super(_Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = utils.positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [_DecoderLayer(d_model, num_heads, dff, rate) 
                                             for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, 
                     look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)    # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class _Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                             target_vocab_size, pe_input, pe_target, rate=0.1):
        super(_Transformer, self).__init__()

        self.encoder = _Encoder(num_layers, d_model, num_heads, dff, 
                                                     input_vocab_size, pe_input, rate)

        self.decoder = _Decoder(num_layers, d_model, num_heads, dff, 
                                                     target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inp, tar, training, enc_padding_mask, 
                     look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)    # (batch_size, inp_seq_len, d_model)
        
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
                tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)    # (batch_size, tar_seq_len, target_vocab_size)
        
        return final_output, attention_weights


class _CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(_CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
