import tensorflow as tf

class URLNet(tf.keras.Model):
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size,
                 word_seq_len, char_seq_len, embedding_size, l2_reg_lambda=0.0,
                 filter_sizes=[3, 4, 5, 6], mode=0):
        super(URLNet, self).__init__()
        self.mode = mode
        self.l2_reg_lambda = l2_reg_lambda
        self.word_seq_len = word_seq_len
        self.char_seq_len = char_seq_len
        self.char_ngram_vocab_size = char_ngram_vocab_size
        self.word_ngram_vocab_size = word_ngram_vocab_size
        self.char_vocab_size = char_vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = 256  # Number of filters per filter size

    def build(self):       
        # Embedding layers
        if self.mode in [4, 5]:
            self.char_embedding = tf.keras.layers.Embedding(
                input_dim=self.char_ngram_vocab_size,  
                output_dim=self.embedding_size,
                name="char_embedding"
            )

        if self.mode in [2, 3, 4, 5]:
            self.word_embedding = tf.keras.layers.Embedding(
                input_dim=self.word_ngram_vocab_size,
                output_dim=self.embedding_size,
                name="word_embedding"
            )

        if self.mode in [1, 3, 5]:
            self.char_seq_embedding = tf.keras.layers.Embedding(
                input_dim=self.char_vocab_size,
                output_dim=self.embedding_size,
                name="char_seq_embedding"
            )

        # Convolutional layers for word embeddings
        if self.mode in [2, 3, 4, 5]:
            self.conv_layers = [
                tf.keras.layers.Conv2D(
                    filters=self.num_filters,
                    kernel_size=(fs, self.embedding_size),
                    activation="relu",
                    name=f"conv_{fs}"
                ) for fs in self.filter_sizes
            ]

        # Convolutional layers for character embeddings
        if self.mode in [1, 3, 5]:
            self.char_conv_layers = [
                tf.keras.layers.Conv2D(
                    filters=self.num_filters,
                    kernel_size=(fs, self.embedding_size),
                    activation="relu",
                    name=f"char_conv_{fs}"
                ) for fs in self.filter_sizes
            ]

        # Fully connected layers
        total_filters = self.num_filters * len(self.filter_sizes)
        if self.mode in [3, 5]:
            fc_input_dim = total_filters * 2  # Concatenated word and char features
        else:
            fc_input_dim = total_filters
            
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.fc1 = tf.keras.layers.Dense(512, activation="relu",
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_lambda))
        self.fc2 = tf.keras.layers.Dense(256, activation="relu",
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_lambda))
        self.fc3 = tf.keras.layers.Dense(128, activation="relu",
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_lambda))
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

        # Call the parent build method
        super().build(self)


    def call(self, inputs, training=False):
        # Apply embeddings
        pooled_outputs = []

        if self.mode in [4, 5]:
            x_char = inputs['input_x_char']
            x_char_pad_idx = inputs['input_x_char_pad_idx']
            embedded_x_char = self.char_embedding(x_char)
            embedded_x_char = tf.multiply(embedded_x_char, x_char_pad_idx)
            sum_ngram_x_char = tf.reduce_sum(embedded_x_char, axis=2)

        if self.mode in [2, 3, 4, 5]:
            x_word = inputs['input_x_word']
            embedded_x_word = self.word_embedding(x_word)

        if self.mode in [1, 3, 5]:
            x_char_seq = inputs['input_x_char_seq']
            embedded_x_char_seq = self.char_seq_embedding(x_char_seq)

        # Combine embeddings for word-level convolution
        if self.mode in [4, 5]:
            sum_ngram_x = sum_ngram_x_char + embedded_x_word
            x_conv_input = tf.expand_dims(sum_ngram_x, -1)
        elif self.mode in [2, 3]:
            x_conv_input = tf.expand_dims(embedded_x_word, -1)

        # Word-level convolution and pooling
        if self.mode in [2, 3, 4, 5]:
            pooled_outputs = []
            for conv_layer in self.conv_layers:
                conv = conv_layer(x_conv_input)
                pool_size = (conv.shape[1], 1)
                pooled = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(1, 1), padding='valid')(conv)
                pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, axis=-1)
            h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)])
            h_pool_flat = self.dropout(h_pool_flat, training=training)

        # Character-level convolution and pooling
        if self.mode in [1, 3, 5]:
            char_x_conv_input = tf.expand_dims(embedded_x_char_seq, -1)
            char_pooled_outputs = []
            for conv_layer in self.char_conv_layers:
                conv = conv_layer(char_x_conv_input)
                pool_size = (conv.shape[1], 1)
                pooled = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(1, 1), padding='valid')(conv)
                char_pooled_outputs.append(pooled)

            h_char_pool = tf.concat(char_pooled_outputs, axis=-1)
            h_char_pool_flat = tf.reshape(h_char_pool, [-1, self.num_filters * len(self.filter_sizes)])
            h_char_pool_flat = self.dropout(h_char_pool_flat, training=training)

        # Combine word and character features
        if self.mode in [3, 5]:
            conv_output = tf.concat([h_pool_flat, h_char_pool_flat], axis=1)
        elif self.mode in [2, 4]:
            conv_output = h_pool_flat
        elif self.mode == 1:
            conv_output = h_char_pool_flat

        # Fully connected layers
        output = self.fc1(conv_output)
        output = self.fc2(output)
        output = self.fc3(output)
        logits = self.output_layer(output)

        return logits