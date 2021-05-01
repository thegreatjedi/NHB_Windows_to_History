from collections import Counter
from itertools import chain, product
from pathlib import Path
from typing import List, Set

import numpy as np
import spacy
import tensorflow as tf
import tensorflow.keras as tfkeras
from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer

from .attn_classes import CNNEncoder, RNNDecoder
from .constants import DECODER_FILE, EMBEDDING_DIM, ENCODER_FILE, GLOVE_FILE, \
    MAX_LEN, TARGET_SIZE, TOKENIZER_FILE, UNITS, VOCAB_SIZE

# Initialise Tensorflow configurations
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

ps = PorterStemmer()
lc = LancasterStemmer()
sb = SnowballStemmer("english")
model = spacy.load('en_core_web_lg')
model.Defaults.stop_words |= {',', '.'}


def build_matrix(word_index, embed_idx, vec_dim):
    emb_mean, emb_std = -0.0033470048, 0.109855264
    embed_matrix = np.random.normal(
        emb_mean, emb_std, (len(word_index) + 1, vec_dim))
    for word, i in word_index.items():
        for candidate in [word, word.lower(), word.upper(), word.capitalize(),
                          ps.stem(word), lc.stem(word), sb.stem(word)]:
            if candidate in embed_idx:
                embed_matrix[i] = embed_idx[candidate]
                break

    return embed_matrix


def decode_image(filename, label=None,
                 image_size=(TARGET_SIZE[0], TARGET_SIZE[1])):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)

    image = (tf.cast(image, tf.float32) / 255.0)
    image = (image - means) / stds  # for qubvel EfficientNet

    image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        return image, label
    

with tf.distribute.get_strategy().scope():
    tokenizer: Tokenizer = tfkeras.preprocessing.text.tokenizer_from_json(
        Path(TOKENIZER_FILE).read_text())
    glove_model = KeyedVectors.load(GLOVE_FILE, mmap='r')
    embedding_matrix = build_matrix(tokenizer.word_index, glove_model,
                                    EMBEDDING_DIM)
    encoder = CNNEncoder(EMBEDDING_DIM)
    encoder.load_weights(ENCODER_FILE)
    decoder = RNNDecoder(embedding_matrix, UNITS, VOCAB_SIZE)
    decoder.load_weights(DECODER_FILE)


def generate_caption(image):
    try:
        hidden = decoder.reset_state(batch_size=1)
    except Exception:
        hidden = decoder.layers[-1].reset_state(batch_size=1)
    
    img_tensor_val = tf.expand_dims(decode_image(image), 0)
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = list()

    for i in range(MAX_LEN):
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            break
        dec_input = tf.expand_dims([predicted_id], 0)

    return ' '.join(
        [word for word in result
         if word != '<start>' and word != '<end>' and word != '<unk>'])


def get_similar_words(captions: List[str], labels: Set[str]) -> Set[str]:
    lbl_doc = model(' '.join(labels))
    cap_doc = [token for token in model(' '.join(captions))
               if not token.is_stop and not token.is_punct
               and (token.pos_ == 'NOUN' or token.pos == 'PROPN'
                    or token.pos == 'ADJECTIVE')]
    filtered_words = [key for key, val
                      in Counter(token.text for token in cap_doc).items()
                      if val > 1]
    cap_doc = [token for token in cap_doc if token.text in filtered_words]
    
    similar_pairs1 = [
        (token1, token2) for token1, token2 in product(cap_doc, lbl_doc)
        if token1 != token2 and token1.similarity(token2) > 0.7
        and token1.similarity(token2) != 1
    ]
    similar_pairs2 = [
        (token1, token2) for token1, token2 in product(cap_doc, cap_doc)
        if token1 != token2 and token1.similarity(token2) > 0.7
        and token1.similarity(token2) != 1
    ]
    
    similar_words1 = {
        token.text for token_pair in similar_pairs1 for token in token_pair
    }
    similar_words2 = {
        token.text for token_pair in similar_pairs2 for token in token_pair
    }
    
    return {word for word in chain(labels, similar_words1, similar_words2)}
