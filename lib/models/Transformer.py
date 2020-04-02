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

        runTransformer(data)

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

    ## Booting function of NN Model + dataset initialization
    def _initArchitecture(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._initArchitecture()")

        vocab_size, max_title_length, all_titles_length, inp_sentences = self._initNNModel(raw_data)
        self._initDataset(raw_data, vocab_size, max_title_length, all_titles_length, inp_sentences)

        return

    ## Booting function of NN Model initialization
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

    ## Loads a model from file.
    def _loadNNModel(self, modelpath):

        return { 'word_model'   :   gensim.models.Word2Vec.load(pt.join(modelpath, "word_model.h5")),
                 'lyric_model'  :   load_model(pt.join(modelpath, "lyric_model.h5"))
               }

    ## Booting function of dataset creation.
    ## Assigns the dataset  to self._dataset
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

    ## Initialize and return word model
    def _initWordModel(self, inp_sentences):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._initWordModel()")
        inp_sentences.append([self._padToken]) # Token that ensembles masking of training weights. Used to pad sequence length
        inp_sentences.append([self._startToken]) # Token that ensembles the start of a sequence
        wm = gensim.models.Word2Vec(inp_sentences, size = 300, min_count = 1, window = 4, iter = 200)
        self._logger.info("Word2Vec word model initialized")
        return wm

    ## Function to initialize and return title model
    ## Needs to be fixed
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

    ## Function to initialize and return lyric model
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

    ## Set class weights for pad token and start token to 0.
    def _setClassWeight(self, vocab_size):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._setClassWeight()")
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

    ## Converts a sentence-list of words-list to a list of chunks of sentences
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
                               else 10 if x == "<ENDLINE>" 
                               else 1 for x in inp] 
                            for inp in encoder_input]

        return encoder_input, decoder_input, decoder_output, song_sample_weight

    def _setClassWeight(self, vocab_size):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._setClassWeight()")
        clw = {}
        for i in range(vocab_size):
            clw[i] = 1
        clw[self.word2idx("<ENDLINE>")] = 50
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

    ## Converts "<ENDLINE>" to '\n' for pretty printing
    ## Also masks meta-tokens but throws a warning
    def _prettyPrint(self, text):
        self._logger.debug("pinkySpeaker.lib.model.Transformer._prettyPrint()")

        if self._startToken in text:
            self._logger.warning("</START> has been found to generated text!")
        if self._padToken in text:
            self._logger.warning("</PAD> has been found to generated text!")
        if "<ENDLINE>" in text:
            self._logger.warning("Endline found in text!")

        return text.replace("<ENDLINE> ", "\n").replace("endfile", "\nEND")

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

        #     if self.idx2word(idx) == "endfile" or (title and self.idx2word(idx) == "<ENDLINE>"):
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

###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################


import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randint

BUFFER_SIZE = 2000
BATCH_SIZE = 64
MAX_LENGTH = 40

tokenizer_pt = None
tokenizer_en = None

def runTransformer(raw_data):

    path = "/home/fivosts/PhD/Code/pinkySpeaker/dataset/pink_floyd"
    temp_dataset = []
    src_dataset = []
    random_lines = []
    for file in os.listdir(path):
        with open(pt.join(path, file), 'r') as f:
            lines = f.readlines()
            lines = [x.replace("\n", "") for x in lines if x.replace("\n", "") != ""]
            random_lines += lines
    # random_lines = [x for sublist in random_lines for x in sublist]

    def labeler(example):
        return example, random_lines[randint(0, len(random_lines) - 1)]

    for file in os.listdir(path):
        line = tf.data.TextLineDataset(pt.join(path, file))
        labelled_line = line.map(lambda key: labeler(key))
        temp_dataset.append(labelled_line)

    global tokenizer_en
    global tokenizer_pt


    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                 (x.numpy() for sublist in temp_dataset for x, _ in sublist), target_vocab_size = 2**13)


    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                    (x.numpy() for sublist in temp_dataset for x, _ in sublist), target_vocab_size = 2**13)

    for file in os.listdir(path):
        line = tf.data.TextLineDataset(pt.join(path, file))
        for l in line:
            if l.numpy().decode("utf-8")  != "":
                # src_dataset.append(([tokenizer_en.vocab_size] + tokenizer_en.encode(l.numpy()) + [tokenizer_en.vocab_size + 1], [tokenizer_en.vocab_size] + tokenizer_en.encode("".join(random_lines[randint(0, len(random_lines) - 1)])) + [tokenizer_en.vocab_size + 1]))
                src_dataset.append(([tokenizer_en.vocab_size] + tokenizer_en.encode(l.numpy()) + [tokenizer_en.vocab_size + 1], [tokenizer_en.vocab_size] + tokenizer_en.encode(l.numpy()) + [tokenizer_en.vocab_size + 1]))

    # examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
    #                                                              as_supervised=True)
    # train_examples1, val_examples1 = examples['train'], examples['validation']

    # train_examples, val_examples = tf.data.Dataset.from_tensor_slices(song_dataset), tf.data.Dataset.from_tensor_slices(song_dataset)

    train_examples = src_dataset
    val_examples = src_dataset

    # print(type(train_examples))
    # print(type(train_examples1))
    # for i in train_examples:
    #     print(type(i))
    #     print(i)
    #     for x in i:
    #         print(type(x))
    #         break
    #     break
    # for i in train_examples1:
    #     print(type(i))
    #     print(i)
    #     for x in i:
    #         print(type(x))
    #         break
    #     break



    sample_string = 'Hello? Is there anybody out there? Is there anyone at home?'

    tokenized_string = tokenizer_en.encode(sample_string)
    print ('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer_en.decode(tokenized_string)
    print ('The original string: {}'.format(original_string))

    assert original_string == sample_string

    for ts in tokenized_string:
        print ('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

    # train_preprocessed = (
    #         tf.convert_to_tensor([x.map(tf_encode) for x in train_examples]))
            #.filter(filter_max_length)
            # cache the dataset to memory to get a speedup while reading from it.
            #.cache()
#            .shuffle(BUFFER_SIZE))
    # val_preprocessed = (
    #          tf.convert_to_tensor([x.map(tf_encode) for x in train_examples]))
    # val_preprocessed = (
    #         val_examples
    #         .map(tf_encode)
    #         .filter(filter_max_length))

    # train_dataset = (train_preprocessed
    #                                  .padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
    #                                  .prefetch(tf.data.experimental.AUTOTUNE))


    # val_dataset = (train_preprocessed
    #                              .padded_batch(BATCH_SIZE,    padded_shapes=([None], [None])))

    # print(tokenizer_pt.vocab_size)
    # print(tokenizer_en.vocab_size)

    train_dataset = train_examples
    val_dataset = train_examples


    # for (batch, (inp, tar)) in enumerate(train_dataset):
    #     print(batch)
    #     print("\n################################\n")
    #     print(inp)
    #     for x, y in zip(inp, tar):
    #         print([t.numpy() for t in x])
    #         print(len([t.numpy() for t in x]))
    #         print(tokenizer_pt.decode(x))
    #         print([t.numpy() for t in y])
    #         print(len([t.numpy() for t in y]))
    #         print(tokenizer_en.decode(y))
    #     break

    # for (batch, (inp, tar)) in enumerate(train_dataset):
    #     for x, y in zip(inp, tar):
    #         print(len([t.numpy() for t in x]))
    #         print(len([t.numpy() for t in y]))


    # train_dataset = (train_preprocessed
    #                                    .padded_batch(BATCH_SIZE)
    #                                    .prefetch(tf.data.experimental.AUTOTUNE))


    # val_dataset = (val_preprocessed
    #                                .padded_batch(BATCH_SIZE))

    print(val_dataset[1])
    pt_batch, en_batch = val_dataset[1]
    pt_batch, en_batch

    pos_encoding = positional_encoding(50, 512)
    print (pos_encoding.shape)

    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()

    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    create_padding_mask(x)

    x = tf.random.uniform((1, 3))
    temp = create_look_ahead_mask(x.shape[1])
    temp

    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10,0,0],
                                                [0,10,0],
                                                [0,0,10],
                                                [0,0,10]], dtype=tf.float32)    # (4, 3)

    temp_v = tf.constant([[     1,0],
                                                [    10,0],
                                                [ 100,5],
                                                [1000,6]], dtype=tf.float32)    # (4, 2)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)    # (1, 3)
    print_out(temp_q, temp_k, temp_v)


    # This query aligns with a repeated key (third and fourth), 
    # so all associated values get averaged.
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)    # (1, 3)
    print_out(temp_q, temp_k, temp_v)


    # This query aligns equally with the first and second key, 
    # so their values get averaged.
    temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)    # (1, 3)
    print_out(temp_q, temp_k, temp_v)



    temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)    # (3, 3)
    print_out(temp_q, temp_k, temp_v)

    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))    # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    out.shape, attn.shape

    sample_ffn = point_wise_feed_forward_network(512, 2048)
    sample_ffn(tf.random.uniform((64, 50, 512))).shape

    sample_encoder_layer = EncoderLayer(512, 8, 2048)

    sample_encoder_layer_output = sample_encoder_layer(
            tf.random.uniform((64, 43, 512)), False, None)

    sample_encoder_layer_output.shape    # (batch_size, input_seq_len, d_model)

    sample_decoder_layer = DecoderLayer(512, 8, 2048)

    sample_decoder_layer_output, _, _ = sample_decoder_layer(
            tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, 
            False, None, None)

    sample_decoder_layer_output.shape    # (batch_size, target_seq_len, d_model)

    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, 
                                                     dff=2048, input_vocab_size=8500,
                                                     maximum_position_encoding=10000)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    print (sample_encoder_output.shape)    # (batch_size, input_seq_len, d_model)

    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, 
                                                     dff=2048, target_vocab_size=8000,
                                                     maximum_position_encoding=5000)
    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input, 
                                                                enc_output=sample_encoder_output, 
                                                                training=False,
                                                                look_ahead_mask=None, 
                                                                padding_mask=None)

    output.shape, attn['decoder_layer2_block2'].shape

    sample_transformer = TFTransformer(
            num_layers=2, d_model=512, num_heads=8, dff=2048, 
            input_vocab_size=tokenizer_en.vocab_size, target_vocab_size=tokenizer_en.vocab_size, 
            pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False, 
                                                                 enc_padding_mask=None, 
                                                                 look_ahead_mask=None,
                                                                 dec_padding_mask=None)

    fn_out.shape    # (batch_size, tar_seq_len, target_vocab_size)

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    dropout_rate = 0.1

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    temp_learning_rate_schedule = CustomSchedule(d_model)

    plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

    transformer = TFTransformer(num_layers, d_model, num_heads, dff,
                                                        input_vocab_size, target_vocab_size, 
                                                        pe_input=input_vocab_size, 
                                                        pe_target=target_vocab_size,
                                                        rate=dropout_rate)

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                                                         optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print ('Latest checkpoint restored!!')

    EPOCHS = 20

    train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp, 
                                                                     True, 
                                                                     enc_padding_mask, 
                                                                     combined_mask, 
                                                                     dec_padding_mask)
            loss = loss_function(tar_real, predictions, loss_object)

        gradients = tape.gradient(loss, transformer.trainable_variables)        
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        train_loss(loss)
        train_accuracy(tar_real, predictions)


    for epoch in range(EPOCHS):

        try:
            start = time.time()
            
            train_loss.reset_states()
            train_accuracy.reset_states()
            
            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(train_dataset):

                # print(len(inp))
                # print(len(tar))
                # print(batch)

                new_inp = tf.convert_to_tensor([[x for x in inp]], dtype = tf.int64)
                new_tar = tf.convert_to_tensor([[x for x in tar]], dtype = tf.int64)
                # print(new_inp)
                # print(new_tar)
                train_step(new_inp, new_tar)
                
                if batch % 50 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                            epoch + 1, batch, train_loss.result(), train_accuracy.result()))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
                
            print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                                                                        train_loss.result(), 
                                                                                                        train_accuracy.result()))

            print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        except KeyboardInterrupt:
            continue


    ## Run 20 times and see how it works....

    for i in range(20):
        line_index = randint(0, len(src_dataset) - 1)
        seed_sentence, real_sentence = src_dataset[line_index]
        seed_sentence = tokenizer_en.decode(seed_sentence[1:-1])
        real_sentence = tokenizer_en.decode(real_sentence[1:-1])
        predicted_sentence = translate(seed_sentence, transformer) ## START and END token
        print("Seed sentence: {}\nReal target: {}\nModel prediction: {}\n".format(seed_sentence, real_sentence, predicted_sentence))


    final_line = randint(0, len(src_dataset) - 1)
    seed_sentence, real_sentence = src_dataset[final_line]
    seed_sentence = tokenizer_en.decode(seed_sentence[1:-1])
    real_sentence = tokenizer_en.decode(real_sentence[1:-1])
    predicted_sentence = translate(seed_sentence, transformer, plot="decoder_layer4_block2") ## START and END token
    print("Seed sentence: {}\nReal target: {}\nModel prediction: {}".format(seed_sentence, real_sentence, predicted_sentence))
    exit(1)

    # translate("este é o primeiro livro que eu fiz.", plot='decoder_layer4_block2')
    # print ("Real translation: this is the first book i've ever done.")

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
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

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

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


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
 
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

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                             maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
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

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                             maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
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


class TFTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                             target_vocab_size, pe_input, pe_target, rate=0.1):
        super(TFTransformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                                                     input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
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


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
            lang1.numpy()) + [tokenizer_pt.vocab_size+1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
            lang2.numpy()) + [tokenizer_en.vocab_size+1]
    
    return lang1, lang2


def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en

def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                                                tf.size(y) <= max_length)




def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                                    np.arange(d_model)[np.newaxis, :],
                                                    d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
    pos_encoding = angle_rads[np.newaxis, ...]
        
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]    # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask    # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
                    to (..., seq_len_q, seq_len_k). Defaults to None.
        
    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)    # (..., seq_len_q, seq_len_k)
    
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)    

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)    # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)    # (..., seq_len_q, depth_v)

    return output, attention_weights




def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
            q, k, v, None)
    print ('Attention weights are:')
    print (temp_attn)
    print ('Output is:')
    print (temp_out)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),    # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)    # (batch_size, seq_len, d_model)
    ])





def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

def evaluate(inp_sentence, transformer):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size + 1]
    
    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
    
    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)
        
    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)
    
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, 
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
        
        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]    # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_en.vocab_size+1:
            return tf.squeeze(output, axis=0), attention_weights
        
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))
    
    sentence = tokenizer_pt.encode(sentence)
    
    attention = tf.squeeze(attention[layer], axis=0)
    
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)
        
        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}
        
        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(result)))
        
        ax.set_ylim(len(result)-1.5, -0.5)
                
        ax.set_xticklabels(
                ['<start>']+[tokenizer_pt.decode([i]) for i in sentence]+['<end>'], 
                fontdict=fontdict, rotation=90)
        
        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result 
                                                if i < tokenizer_en.vocab_size], 
                                             fontdict=fontdict)
        
        ax.set_xlabel('Head {}'.format(head+1))
    
    plt.tight_layout()
    plt.show()


def translate(sentence, transformer, plot=''):
    result, attention_weights = evaluate(sentence, transformer)
    
    predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])    

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)
    return predicted_sentence
