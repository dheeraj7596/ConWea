import keras
from keras.layers import (
    Dense, GRU, TimeDistributed, Input,
    Embedding, Bidirectional, Lambda
)
from keras.models import Model
from keras_han.layers import AttentionLayer


class HAN(Model):
    def __init__(
            self, max_words, max_sentences, output_size,
            embedding_matrix, word_encoding_dim=200,
            sentence_encoding_dim=200, inputs=None,
            outputs=None, name='han-for-docla'
    ):
        """
        A Keras implementation of Hierarchical Attention networks
        for document classification.
        :param max_words: The maximum number of words per sentence
        :param max_sentences: The maximum number of sentences
        :param output_size: The dimension of the last layer (i.e.
            the number of classes you wish to predict)
        :param embedding_matrix: The embedding matrix to use for
            representing words
        :param word_encoding_dim: The dimension of the GRU
            layer in the word encoder.
        :param sentence_encoding_dim: The dimension of the GRU
            layer in the sentence encoder.
        """
        self.max_words = max_words
        self.max_sentences = max_sentences
        self.output_size = output_size
        self.embedding_matrix = embedding_matrix
        self.word_encoding_dim = word_encoding_dim
        self.sentence_encoding_dim = sentence_encoding_dim


        in_tensor, out_tensor = self._build_network()

        super(HAN, self).__init__(
            inputs=in_tensor, outputs=out_tensor, name=name
        )

    def build_word_encoder(self, max_words, embedding_matrix, encoding_dim=200):
        """
        Build the model that embeds and encodes in context the
        words used in a sentence. The return model takes a tensor of shape
        (batch_size, max_length) that represents a collection of sentences
        and returns an encoded representation of these sentences.
        :param max_words: (int) The maximum sentence length this model accepts
        :param embedding_matrix: (2d array-like) A matrix with the i-th row
            representing the embedding of the word represented by index i.
        :param encoding_dim: (int, should be even) The dimension of the
            bidirectional encoding layer. Half of the nodes are used in the
            forward direction and half in the backward direction.
        :return: Instance of keras.Model
        """
        assert encoding_dim % 2 == 0, "Embedding dimension should be even"

        vocabulary_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        embedding_layer = Embedding(
            vocabulary_size, embedding_dim,
            weights=[embedding_matrix], input_length=max_words,
            trainable=False
        )

        sentence_input = Input(shape=(max_words,), dtype='int32')
        embedded_sentences = embedding_layer(sentence_input)
        encoded_sentences = Bidirectional(
            GRU(int(encoding_dim / 2), return_sequences=True)
        )(embedded_sentences)

        return Model(
            inputs=[sentence_input], outputs=[encoded_sentences], name='word_encoder'
        )

    def build_sentence_encoder(self, max_sentences, summary_dim, encoding_dim=200):
        """
        Build the encoder that encodes the vector representation of
        sentences in their context.
        :param max_sentences: The maximum number of sentences that can be
            passed. Use zero-padding to supply shorter sentences.
        :param summary_dim: (int) The dimension of the vectors that summarizes
            sentences. Should be equal to the encoding_dim of the word
            encoder.
        :param encoding_dim: (int, even) The dimension of the vector that
            summarizes sentences in context. Half is used in forward direction,
            half in backward direction.
        :return: Instance of keras.Model
        """
        assert encoding_dim % 2 == 0, "Embedding dimension should be even"

        text_input = Input(shape=(max_sentences, summary_dim))
        encoded_sentences = Bidirectional(
            GRU(int(encoding_dim / 2), return_sequences=True)
        )(text_input)
        return Model(
            inputs=[text_input], outputs=[encoded_sentences], name='sentence_encoder'
        )

    def _build_network(self):
        """
        Build the graph that represents this network
        :return: in_tensor, out_tensor, Tensors representing the input and output
            of this network.
        """
        in_tensor = Input(shape=(self.max_sentences, self.max_words))

        word_encoder = self.build_word_encoder(
            self.max_words, self.embedding_matrix, self.word_encoding_dim
        )

        word_rep = TimeDistributed(
            word_encoder, name='word_encoder'
        )(in_tensor)

        # Sentence Rep is a 3d-tensor (batch_size, max_sentences, word_encoding_dim)
        sentence_rep = TimeDistributed(
            AttentionLayer(), name='word_attention'
        )(word_rep)

        doc_rep = self.build_sentence_encoder(
            self.max_sentences, self.word_encoding_dim, self.sentence_encoding_dim
        )(sentence_rep)

        # We get the final representation by applying our attention mechanism
        # to the encoded sentences
        doc_summary = AttentionLayer(name='sentence_attention')(doc_rep)

        out_tensor = Dense(
            self.output_size, activation='softmax', name='class_prediction'
        )(doc_summary)

        return in_tensor, out_tensor

    def get_config(self):
        config = {
            'max_words': self.max_words,
            'max_sentences': self.max_sentences,
            'output_size': self.output_size,
            'embedding_matrix': self.embedding_matrix,
            'word_encoding_dim': self.word_encoding_dim,
            'sentence_encoding_dim': self.sentence_encoding_dim,
            'base_config': super(HAN, self).get_config()
        }

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        Keras' API isn't really extendible at this point
        therefore we need to use a bit hacky solution to
        be able to correctly reconstruct the HAN model
        from a config. This therefore does not reconstruct
        a instance of HAN model, but actually a standard
        Keras model that behaves exactly the same.
        """
        base_config = config.pop('base_config')

        return Model.from_config(
            base_config, custom_objects=custom_objects
        )

    def predict_sentence_attention(self, X):
        """
        For a given set of texts predict the attention
        weights for each sentence.
        :param X: 3d-tensor, similar to the input for predict
        :return: 2d array (num_obs, max_sentences) containing
            the attention weights for each sentence
        """
        att_layer = self.get_layer('sentence_attention')
        prev_tensor = att_layer.input

        # Create a temporary dummy layer to hold the
        # attention weights tensor
        dummy_layer = Lambda(
            lambda x: att_layer._get_attention_weights(x)
        )(prev_tensor)

        return Model(self.input, dummy_layer).predict(X)

