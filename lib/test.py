import numpy as np
from keras_transformer import get_model
from keras_transformer import decode

# Build a small toy token dictionary
tokens = 'The dominant sequence transduction models are based on complex recurrent or convolutional\
            neural networks in an encoder-decoder configuration. The best performing models also connect\
            the encoder and decoder through an attention mechanism. We propose a new simple network architecture,\
            the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.\
            Experiments on two machine translation tasks show these models to be superior in quality while being more \
            parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT \
            2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. \
            On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU \
            score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models \
            from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English \
            constituency parsing both with large and limited training data. '.replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .replace("  ", " ")\
                                                                             .split(' ')





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
    # encode_tokens = list(map(lambda x: token_dict[x], encode_tokens))
    # decode_tokens = list(map(lambda x: token_dict[x], decode_tokens))
    # output_tokens = list(map(lambda x: [token_dict[x]], output_tokens))
    encoder_inputs_no_padding.append(encode_tokens[:i + 2])
    encoder_inputs.append(encode_tokens)
    decoder_inputs.append(decode_tokens)
    decoder_outputs.append(output_tokens)

# Build the model
model = get_model(
    token_num=len(token_dict),
    embed_dim=300,
    encoder_num=5,
    decoder_num=4,
    head_num=6,
    hidden_dim=256,
    attention_activation='relu',
    feed_forward_activation='relu',
    dropout_rate=0.05,
    embed_weights=np.random.random((122, 300)),
)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)
model.summary()

print("Vocab size: {}".format(len(token_dict)))
print("Seq size: {}".format(len(tokens)))
print("Encoder input shape: {}".format(np.asarray(encoder_inputs).shape))
print("Decoder input shape: {}".format(np.asarray(decoder_inputs).shape))
print("Decoder output shape: {}".format(np.asarray(decoder_outputs).shape))
print("Enc for prediction: {}".format(np.asarray(encoder_inputs_no_padding).shape))

for x, y, z in zip(encoder_inputs, decoder_inputs, decoder_outputs):
    print(x)
    print(y)
    print(z)
    print("\n\n\n\n")

# Train the model
model.fit(
    x=[np.asarray(encoder_inputs), np.asarray(decoder_inputs)],
    y=np.asarray(decoder_outputs),
    epochs=70,
    batch_size = 4
)


print(encoder_inputs_no_padding)

decoded = decode(
    model,
    encoder_inputs_no_padding,
    start_token=token_dict['<START>'],
    end_token=token_dict['<END>'],
    pad_token=token_dict['<PAD>'],
    max_len=200,
)
print(decoded)
token_dict_rev = {v: k for k, v in token_dict.items()}
for i in range(len(decoded)):
    print("SEQ START")
    print(' '.join(map(lambda x: token_dict_rev[x], encoder_inputs_no_padding[i])))
    print(' '.join(map(lambda x: token_dict_rev[x], decoded[i])))
    print(decoded[i])
    print("SEQ END\n\n")

