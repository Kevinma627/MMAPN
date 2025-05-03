import tensorflow as tf

class HTMLNet(tf.keras.Model):
    def __init__(self, word_ngram_vocab_size, char_vocab_size,
                 word_seq_len, char_seq_len, embedding_size, l2_reg_lambda=0.0,
                 filter_sizes=[3, 4, 5, 6, 7, 8, 9, 10]): # HTMLPhish paper proposed 8 kernel sizes
        
        super(HTMLNet, self).__init__()
        self.word_seq_len = word_seq_len
        self.char_seq_len = char_seq_len
        self.word_ngram_vocab_size = word_ngram_vocab_size
        self.char_vocab_size = char_vocab_size
        self.embedding_size = embedding_size
        self.l2_reg_lambda = l2_reg_lambda
        self.filter_sizes = filter_sizes
        self.num_filters = 256 # Increased filters because model has trouble extracting features

    def build(self):       
        # Embedding layers
        self.word_embedding = tf.keras.layers.Embedding(
            input_dim=self.word_ngram_vocab_size,
            output_dim=self.embedding_size,
            name="word_embedding"
        )
        self.char_seq_embedding = tf.keras.layers.Embedding(
            input_dim=self.char_vocab_size,
            output_dim=self.embedding_size,
            name="char_seq_embedding"
        )

        # Convolutional layers
        self.conv_layers = [
            tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=(fs, self.embedding_size),
                activation="relu",
                name=f"conv_{fs}"
            ) for fs in self.filter_sizes
        ]
        
        # Dense layers
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.fc1 = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(negative_slope=0.1), 
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_lambda))
        self.fc2 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(negative_slope=0.1),
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_lambda))
        self.fc3 = tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(negative_slope=0.1),
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_lambda))
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

        # Call the parent build method
        super().build(self)

    def call(self, inputs, training=False):
        # Apply embeddings
        x_word = inputs['input_x_word']
        embedded_x_word = self.word_embedding(x_word)
        x_char_seq = inputs['input_x_char_seq']
        embedded_x_char_seq = self.char_seq_embedding(x_char_seq)

        embedded_x_word = tf.expand_dims(embedded_x_word, -1)
        embedded_x_char_seq = tf.expand_dims(embedded_x_char_seq, -1) 

        # Concat char and word representations
        x_conv_input = tf.concat([embedded_x_char_seq, embedded_x_word],axis=1)

        # Conv layers and pooling
        pooled_outputs = []
        for conv_layer in self.conv_layers:
            conv = conv_layer(x_conv_input)
            pooled = tf.keras.layers.GlobalMaxPooling2D()(conv)
            pooled_outputs.append(pooled)

        h_pool = tf.concat(pooled_outputs, axis=-1)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)])

        # Dense layer
        dense_output = self.fc1(h_pool_flat)
        dense_output = self.fc2(dense_output)
        dense_output = self.fc3(dense_output)

        # Dropout
        output = self.dropout(dense_output, training=training)

        # Output layer
        logits = self.output_layer(output)

        return logits