import argparse
from tqdm import tqdm
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from utils import *
import pickle
from URLNet import URLNet
import os

def main():
    parser = argparse.ArgumentParser(description="Test URLNet model")

    # Data arguments
    parser.add_argument('--data_data_dir', type=str, default='Test_Data/test_all.txt', help="Location of data file")
    parser.add_argument('--data_max_len_words', type=int, default=200, help="Maximum length of URL in words")
    parser.add_argument('--data_max_len_chars', type=int, default=200, help="Maximum length of URL in characters")
    parser.add_argument('--data_max_len_subwords', type=int, default=20, help="Maximum length of word in subwords/characters")
    parser.add_argument('--data_delimit_mode', type=int, default=1, help="0: special chars, 1: special chars + each char as word")

    # Model arguments
    parser.add_argument('--model_emb_dim', type=int, default=32, help="Embedding dimension size")
    parser.add_argument('--model_filter_sizes', type=str, default="3,4,5,6", help="Filter sizes of the convolution layer")
    parser.add_argument('--model_emb_mode', type=int, default=3, help="1: charCNN, 2: wordCNN, etc.")

    # Test arguments
    parser.add_argument('--test_batch_size', type=int, default=128, help="Batch size")

    # Log arguments
    parser.add_argument('--log_output_dir', type=str, default="Model/runs_url", help="Output directory")
    parser.add_argument('--log_checkpoint_dir', type=str, default="Model/runs_url", help="Checkpoint directory")

    args = parser.parse_args()

    for key, val in vars(args).items():
        print(f"{key}={val}")

    # Load data
    urls, labels = read_data(args.data_data_dir)

    # Remove protocol/subdomain if necessary (increases uniqueness in data)
    for i in range(len(urls)):
        if urls[i].startswith("https://www."):
            urls[i] = urls[i][12:]
        elif urls[i].startswith("https://"):
            urls[i] = urls[i][8:]

    x, word_reverse_dict = get_word_vocab(urls, args.data_max_len_words)
    word_x = get_words(x, word_reverse_dict, args.data_delimit_mode, urls)

    # Load dictionaries
    ngram_dict = pickle.load(open(args.log_checkpoint_dir + "/subwords_dict.p", "rb"))
    print("Size of subword vocabulary (train): {}".format(len(ngram_dict)))
    word_dict = pickle.load(open(args.log_checkpoint_dir + "/words_dict.p", "rb"))
    print("Size of word vocabulary (train): {}".format(len(word_dict)))
    chars_dict = pickle.load(open(args.log_checkpoint_dir + "/chars_dict.p", "rb"))
    print("Size of character vocabulary (train): {}".format(len(chars_dict)))
    ngrams_dict = ngram_dict

    ngramed_id_x, worded_id_x = ngram_id_x_from_dict(word_x, args.data_max_len_subwords, ngram_dict, word_dict)
    chared_id_x = char_id_x(urls, chars_dict, args.data_max_len_chars)

    print("Number of testing URLs: {}".format(len(labels)))

    # Pad sequences
    x_char_seq_padded = pad_sequences(chared_id_x, maxlen=args.data_max_len_chars, padding='post')
    x_word_padded = pad_sequences(worded_id_x, maxlen=args.data_max_len_words, padding='post')

    # Prepare datasets
    def make_batches(batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((x_char_seq_padded, x_word_padded, labels))
        dataset = dataset.batch(batch_size)
        return dataset

    def prep_batches(batch):
        x_char_seq, x_word, y_batch = batch
        x_batch_list = [x_char_seq, x_word]
        y_batch = tf.cast(y_batch, tf.float32)
        y_batch = tf.reshape(y_batch, (-1, 1))
        return x_batch_list, y_batch

    # Load the model
    cnn = URLNet(
        char_ngram_vocab_size=len(ngrams_dict) + 1,
        word_ngram_vocab_size=len(word_dict) + 1,
        char_vocab_size=len(chars_dict) + 1,
        embedding_size=args.model_emb_dim,
        word_seq_len=args.data_max_len_words,
        char_seq_len=args.data_max_len_chars,
        mode=args.model_emb_mode,
        filter_sizes=list(map(int, args.model_filter_sizes.split(","))),
    )

    # Restore the checkpoint
    checkpoint = tf.train.Checkpoint(model=cnn)
    latest_checkpoint = tf.train.latest_checkpoint(args.log_checkpoint_dir + "/checkpoints")
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint).expect_partial()
        print("Restored from {}".format(latest_checkpoint))
    else:
        print("No checkpoint found at {}".format(args.log_checkpoint_dir))
        return

    # Prepare dataset
    test_dataset = make_batches(args.test_batch_size)

    # Evaluation
    all_predictions = []
    all_scores = []

    for batch in tqdm(test_dataset, desc="Testing"):
        x_batch_list, y_batch = prep_batches(batch)
        inputs = {'input_x_char_seq': x_batch_list[0], 'input_x_word': x_batch_list[1]}

        logits = cnn(inputs, training=False)
        predictions = (logits.numpy() > 0.72).astype(int).flatten() # Adjust prediction threshold accordingly
        scores = logits.numpy().flatten()
        all_predictions.extend(predictions)
        all_scores.extend(scores)

    # Save test results
    output_file = os.path.join(args.log_output_dir, "test_results.txt")
    save_test_result(labels, all_predictions, all_scores, output_file)

if __name__ == "__main__":
    main()