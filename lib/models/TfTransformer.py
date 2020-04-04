#!/usr/bin/env python
import sys
from os import path as pt
from os import makedirs
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.native import logger as l
from eupy.native import plotter as plt

from lib import history

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import time

class TfTransformer:

    def __init__(self, data = None, model = None, batch_size = 4, LSTM_Depth = 3, sequence_length = 30):
        self._logger = l.getLogger()
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer.__init__()")

        self._num_layers = 2
        self._d_model = 64
        self._dff = 64
        self._num_heads = 2

        ## Training history object
        self._history = history.history("TfTransformer", num_layers = self._num_layers,
                                                         d_model = self._d_model,
                                                         dff = self._dff,
                                                         num_heads = self._num_heads,
                                        )
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
        self._initNNModel(raw_data)
        self._logger.info("Transformer architecture initialized")
        return

    ## Booting function of NN Model initialization
    def _initNNModel(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer._initNNModel()")
        self._logger.info("Initialize NN Model")

        input_vocab_size = self._model['tokenizer'].vocab_size + 2
        target_vocab_size = self._model['tokenizer'].vocab_size + 2
        dropout_rate = 0.1
        ##

        self._model['transformer'] = self._setupTransformer(self._num_layers,
                                                            self._d_model, 
                                                            self._dff, 
                                                            self._num_heads, 
                                                            input_vocab_size, 
                                                            target_vocab_size, 
                                                            dropout_rate
                                                            )
        self._model['optimizer'] = self._setupOptimizer(self._d_model)

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

    ## Just fit it!
    def fit(self, epochs = 200, live_plot = True, save_model = None):
        self._logger.debug("pinkySpeaker.lib.model.TfTransformer.fit()")

        # checkpoint_path = "./checkpoints/train"
        ## TODO checkout checkpoint manager
        # ckpt = tf.train.Checkpoint(transformer=transformer,
        #                                                      optimizer=optimizer)

        # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        # if ckpt_manager.latest_checkpoint:
        #     ckpt.restore(ckpt_manager.latest_checkpoint)
        #     print ('Latest checkpoint restored!!')
        if live_plot:
            ylim = 0
            plotted_data = {
                'loss': {'y': []},
                'accuracy': {'y': []}
            }

        for epoch in range(epochs):
            start = time.time()
            
            self._model['optimizer']['loss'].reset_states()
            self._model['optimizer']['accuracy'].reset_states()
            
            for (batch, (inp, tar)) in enumerate(self._dataset):
                self.train_step(inp, tar)
        
                if batch % 5 == 0:
                    self._logger.info('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'
                            .format( epoch + 1, batch, self._model['optimizer']['loss'].result(), 
                                    self._model['optimizer']['accuracy'].result())
                                    )

            self._history.loss = self._model['optimizer']['loss'].result().numpy()
            self._history.accuracy = self._model['optimizer']['accuracy'].result().numpy()

            if live_plot:
                ylim = max(ylim, self._model['optimizer']['loss'].result().numpy())
                plotted_data['loss']['y'] = self._history.loss
                plotted_data['accuracy']['y'] = self._history.accuracy
                plt.linesSingleAxis(plotted_data, y_label = ("Loss vs Accuracy", 13), 
                                                  x_label = ("Epochs", 13), 
                                                  vert_grid = True,
                                                  plot_title = self._history.modeltype + self._history.properties,
                                                  y_lim = ylim + 0.1*ylim, x_lim = epochs, 
                                                  live = True)



            # if (epoch + 1) % 5 == 0:
            #     ckpt_save_path = ckpt_manager.save()
            #     print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
                
            self._logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                                self._model['optimizer']['loss'].result(), 
                                                                self._model['optimizer']['accuracy'].result())
                                                                )
            self._logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        return 

    @tf.function(input_signature = [tf.TensorSpec(shape = (None, None), dtype = tf.int64),
                                    tf.TensorSpec(shape = (None, None), dtype = tf.int64)]
                )
    def train_step(self, inp, target):

        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]
        
        enc_padding_mask, combined_mask, dec_padding_mask = utils().create_masks(inp, tar_inp)
        
        with tf.GradientTape() as tape:
            predictions, _ = self._model['transformer'](inp, tar_inp, 
                                                         True, 
                                                         enc_padding_mask, 
                                                         combined_mask, 
                                                         dec_padding_mask)
            loss = utils().loss_function(tar_real, predictions, self._model['optimizer']['loss_obj'])

        gradients = tape.gradient(loss, self._model['transformer'].trainable_variables)        
        self._model['optimizer']['adam'].apply_gradients(zip(gradients, self._model['transformer'].trainable_variables))
        
        self._model['optimizer']['loss'](loss)
        self._model['optimizer']['accuracy'](tar_real, predictions)

        return

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
        scaled_attention, attention_weights = utils().scaled_dot_product_attention(
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
        self.ffn = utils().point_wise_feed_forward_network(d_model, dff)

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

        self.ffn = utils().point_wise_feed_forward_network(d_model, dff)
 
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
        self.pos_encoding = utils().positional_encoding(maximum_position_encoding, d_model)
        
        
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
        self.pos_encoding = utils().positional_encoding(maximum_position_encoding, d_model)
        
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

class utils:
    def __init__(self):
        return

    def filter_max_length(self, x, y, max_length=40):
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                                        np.arange(d_model)[np.newaxis, :],
                                                        d_model)
        
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]    # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask    # (seq_len, seq_len)

    def scaled_dot_product_attention(self, q, k, v, mask):
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

    def print_out(self, q, k, v):
        temp_out, temp_attn = self.scaled_dot_product_attention(
                q, k, v, None)
        l.getLogger().info('Attention weights are:')
        l.getLogger().info(temp_attn)
        l.getLogger().info('Output is:')
        l.getLogger().info(temp_out)
        return

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
                tf.keras.layers.Dense(dff, activation='relu'),    # (batch_size, seq_len, dff)
                tf.keras.layers.Dense(d_model)    # (batch_size, seq_len, d_model)
        ])

    def loss_function(self, real, pred, loss_object):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)
        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return enc_padding_mask, combined_mask, dec_padding_mask

    ## TODO move this to class
    ## Too many deps here. This won't work
    def evaluate(self, inp_sentence, transformer):
        start_token = [tokenizer.vocab_size]
        end_token = [tokenizer.vocab_size + 1]
        
        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + tokenizer.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)
        
        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [tokenizer.vocab_size]
        output = tf.expand_dims(decoder_input, 0)
            
        MAX_LENGTH = 40
        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
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
            if predicted_id == tokenizer.vocab_size+1:
                return tf.squeeze(output, axis=0), attention_weights
            
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    ## TODO move this to plotter
    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))
        
        sentence = tokenizer.encode(sentence)
        
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
                    ['<start>']+[tokenizer.decode([i]) for i in sentence]+['<end>'], 
                    fontdict=fontdict, rotation=90)
            
            ax.set_yticklabels([tokenizer.decode([i]) for i in result 
                                                    if i < tokenizer.vocab_size], 
                                                 fontdict=fontdict)
            
            ax.set_xlabel('Head {}'.format(head+1))
        
        plt.tight_layout()
        plt.show()

    ## TODO rename this to predict and move to tf.transformer
    def translate(self, sentence, transformer, plot=''):
        result, attention_weights = self.evaluate(sentence, transformer)
        predicted_sentence = tokenizer.decode([i for i in result if i < tokenizer.vocab_size])    

        if plot:
            self.plot_attention_weights(attention_weights, sentence, result, plot)
        return predicted_sentence