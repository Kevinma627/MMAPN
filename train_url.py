import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from URLNet import URLNet
from utils import *
import pickle
import os
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Train URLNet model")

    # Data arguments
    parser.add_argument('--data_data_dir', type=str, default='Train_Data/train_url.txt', help="Location of data file")
    parser.add_argument('--data_max_len_words', type=int, default=200, help="Maximum length of URL in words")
    parser.add_argument('--data_max_len_chars', type=int, default=200, help="Maximum length of URL in characters")
    parser.add_argument('--data_max_len_subwords', type=int, default=20, help="Maximum length of word in subwords/characters")
    parser.add_argument('--data_min_word_freq', type=int, default=1, help="Minimum frequency of word to build vocabulary")
    parser.add_argument('--data_dev_pct', type=float, default=0.2, help="Percentage of data used for validation")
    parser.add_argument('--data_delimit_mode', type=int, default=1, help="0: special chars, 1: special chars + each char as word")

    # Model arguments
    parser.add_argument('--model_emb_dim', type=int, default=32, help="Embedding dimension size")
    parser.add_argument('--model_filter_sizes', type=str, default="3,4,5,6", help="Filter sizes of the convolution layer")
    parser.add_argument('--model_emb_mode', type=int, default=3, help="1: charCNN, 2: wordCNN, etc.")

    # Training arguments
    parser.add_argument('--train_nb_epochs', type=int, default=2, help="Number of training epochs")
    parser.add_argument('--train_batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--train_l2_reg_lambda', type=float, default=0.0, help="L2 regularization lambda")
    parser.add_argument('--train_lr', type=float, default=0.001, help="Learning rate")

    # Logging arguments
    parser.add_argument('--log_output_dir', type=str, default="Model/runs_url", help="Output directory")

    args = parser.parse_args()

    for key, val in vars(args).items():
        print(f"{key}={val}")

    # Data preparation
    urls, labels = read_data(args.data_data_dir)

    # Remove protocol/subdomain if necessary (increases uniqueness in data)
    # If URL starts with http not https that is still taken into account (http is not removed)
    for i in range(len(urls)):
        if urls[i].startswith("https://www."):
            urls[i] = urls[i][12:]
        elif urls[i].startswith("https://"):
            urls[i] = urls[i][8:]

    high_freq_words = None
    if args.data_min_word_freq > 1:
        x1, word_reverse_dict = get_word_vocab(urls, args.data_max_len_words, args.data_min_word_freq)
        high_freq_words = sorted(list(word_reverse_dict.values()))
        print(f"Number of words with freq >= {args.data_min_word_freq}: {len(high_freq_words)}")

    x, word_reverse_dict = get_word_vocab(urls, args.data_max_len_words)
    word_x = get_words(x, word_reverse_dict, args.data_delimit_mode, urls)
    ngramed_id_x, ngrams_dict, worded_id_x, words_dict = ngram_id_x(word_x, args.data_max_len_subwords, high_freq_words)

    chars_dict = ngrams_dict
    chared_id_x = char_id_x(urls, chars_dict, args.data_max_len_chars)

    print(f"Size of ngrams_dict: {len(ngrams_dict)}")
    print(f"Size of words_dict: {len(words_dict)}")
    print(f"Size of chars_dict: {len(chars_dict)}")

    # Split data into training and validation sets using stratification
    indices = np.arange(len(labels))
    x_train_indices, x_val_indices, y_train, y_val = train_test_split(
        indices, labels, test_size=args.data_dev_pct, random_state=42, stratify=labels)

    # Get corresponding data
    x_train_char_seq = [chared_id_x[i] for i in x_train_indices]
    x_val_char_seq = [chared_id_x[i] for i in x_val_indices]
    x_train_word = [worded_id_x[i] for i in x_train_indices]
    x_val_word = [worded_id_x[i] for i in x_val_indices]

    # Pad sequences
    x_train_char_seq_padded = pad_sequences(x_train_char_seq, maxlen=args.data_max_len_chars, padding='post')
    x_val_char_seq_padded = pad_sequences(x_val_char_seq, maxlen=args.data_max_len_chars, padding='post')
    x_train_word_padded = pad_sequences(x_train_word, maxlen=args.data_max_len_words, padding='post')
    x_val_word_padded = pad_sequences(x_val_word, args.data_max_len_words, padding='post')

    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Instantiate the model
    cnn = URLNet(
        char_ngram_vocab_size=len(ngrams_dict) + 1,
        word_ngram_vocab_size=len(words_dict) + 1,
        char_vocab_size=len(chars_dict) + 1,
        embedding_size=args.model_emb_dim,
        word_seq_len=args.data_max_len_words,
        char_seq_len=args.data_max_len_chars,
        l2_reg_lambda=args.train_l2_reg_lambda,
        mode=args.model_emb_mode,
        filter_sizes=list(map(int, args.model_filter_sizes.split(","))),
    #    num_classes=1,  # Binary classification
    )

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.train_lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # Prepare datasets
    train_dataset = make_batches(x_train_char_seq_padded, x_train_word_padded, y_train, args.train_batch_size, shuffle=True)
    val_dataset = make_batches(x_val_char_seq_padded, x_val_word_padded, y_val, args.train_batch_size, shuffle=False)

    # Define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

    # Checkpoint setup
    checkpoint_dir = os.path.join(args.log_output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=cnn)

    # Save dictionaries
    if not os.path.exists(args.log_output_dir):
        os.makedirs(args.log_output_dir)
    with open(os.path.join(args.log_output_dir, "subwords_dict.p"), "wb") as f:
        pickle.dump(ngrams_dict, f)
    with open(os.path.join(args.log_output_dir, "words_dict.p"), "wb") as f:
        pickle.dump(words_dict, f)
    with open(os.path.join(args.log_output_dir, "chars_dict.p"), "wb") as f:
        pickle.dump(chars_dict, f)

    # Training loop
    for epoch in range(args.train_nb_epochs):
        print(f"\nStart of epoch {epoch+1}")
        train_loss.reset_state()
        train_accuracy.reset_state()

        for batch in tqdm(train_dataset, desc="Training"):
            x_batch_list, y_batch = prep_batches(batch)
            inputs = {'input_x_char_seq': x_batch_list[0], 'input_x_word': x_batch_list[1]}  # For emb_mode=3

            with tf.GradientTape() as tape:
                logits = cnn(inputs, training=True)
                loss = loss_fn(y_batch, logits)

            gradients = tape.gradient(loss, cnn.trainable_variables)
            optimizer.apply_gradients(zip(gradients, cnn.trainable_variables))

            train_loss(loss)
            train_accuracy(y_batch, logits)

        print(f"Epoch {epoch+1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}")

        # Validation
        val_loss.reset_state()
        val_accuracy.reset_state()

        for batch in val_dataset:
            x_batch_list, y_batch = prep_batches(batch)
            inputs = {'input_x_char_seq': x_batch_list[0], 'input_x_word': x_batch_list[1]}  # For emb_mode=3

            logits = cnn(inputs, training=False)
            loss = loss_fn(y_batch, logits)

            val_loss(loss)
            val_accuracy(y_batch, logits)

        print(f"Validation Loss: {val_loss.result()}, Validation Accuracy: {val_accuracy.result()}")

        # Save checkpoint
        checkpoint.save(file_prefix=checkpoint_prefix)

if __name__ == "__main__":
    main()