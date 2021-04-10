import efficientnet.tfkeras as efn
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Embedding, GRU
from .constants import TARGET_SIZE


class BahdanauAttention(Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape
        #     == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to
        # self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights


class CNNEncoder(Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNNEncoder, self).__init__()
        self.cnn0 = efn.EfficientNetB3(
            weights='noisy-student', input_shape=TARGET_SIZE, include_top=False
        )
        # e.g. layers[-1].output = TensorShape([None, 10, 10, 1536]) for
        # B3 (not global pooling)
        self.cnn = Model(self.cnn0.input, self.cnn0.layers[-1].output)
        self.cnn.trainable = False
        # shape after fc == (batch_size, attention_features_shape,
        # embedding_dim) >> this is my mistake, should be hidden instead
        # of embedding_dim
        self.fc = Dense(embedding_dim)
    
    # here, x is img-tensor of target_size
    def call(self, x):
        x = self.cnn(x)  # 4D
        x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))  # 3D
        x = self.fc(x)
        x = tf.nn.relu(x)
        
        return x
    
    
class RNNDecoder(Model):
    def __init__(self, embedding_matrix, units, vocab_size):
        super(RNNDecoder, self).__init__()
        self.units = units
        self.vocab_size = embedding_matrix.shape[0]
        # new interface of pretrained embedding weights:
        # https://github.com/tensorflow/tensorflow/issues/31086
        # see also: https://stackoverflow.com/questions/55770009/how-to-use-a-pre-trained-embedding-matrix-in-tensorflow-2-0-rnn-as-initial-weigh
        self.embedding = Embedding(
            self.vocab_size, embedding_matrix.shape[1],
            embeddings_initializer=Constant(embedding_matrix), trainable=False,
            mask_zero=True)
        self.gru = GRU(
            self.units, return_sequences=True, return_state=True,
            recurrent_initializer='glorot_uniform')
        self.fc1 = Dense(self.units)
        self.fc2 = Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    # x=sequence of words
    # features=image's extracted features
    # hidden=GRU's hidden unit
    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        # x shape after passing through embedding
        #     == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (
        #     batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)
        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))
        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)
        
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
