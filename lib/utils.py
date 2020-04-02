#!/usr/bin/env python

import tensorflow as tf
import sys
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.native import logger as l
import numpy as np

def filter_max_length(x, y, max_length=40):
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
    l.getLogger().info('Attention weights are:')
    l.getLogger().info(temp_attn)
    l.getLogger().info('Output is:')
    l.getLogger().info(temp_out)
    return

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
    start_token = [tokenizer.vocab_size]
    end_token = [tokenizer.vocab_size + 1]
    
    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
    
    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer.vocab_size]
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
        if predicted_id == tokenizer.vocab_size+1:
            return tf.squeeze(output, axis=0), attention_weights
        
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

## TODO move this to plotter
def plot_attention_weights(attention, sentence, result, layer):
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
def translate(sentence, transformer, plot=''):
    result, attention_weights = evaluate(sentence, transformer)
    predicted_sentence = tokenizer.decode([i for i in result if i < tokenizer.vocab_size])    

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)
    return predicted_sentence