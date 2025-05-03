import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from utils import *
import pickle
from HTMLNet import HTMLNet
import os

def main():
    parser = argparse.ArgumentParser(description="Test HTMLNet model")

    # Data arguments
    parser.add_argument('--data_data_dir', type=str, default='data/train_converted.txt', help="Location of data file")
    parser.add_argument('--data_max_len_words', type=int, default=200, help="Maximum length of URL in words")
    parser.add_argument('--data_max_len_chars', type=int, default=200, help="Maximum length of URL in characters")
    parser.add_argument('--data_max_len_subwords', type=int, default=20, help="Maximum length of word in subwords/characters")
    parser.add_argument('--data_subword_dict_dir', type=str, default="Model/runs_html/subwords_dict.p", help="Directory of subword dictionary")
    parser.add_argument('--data_word_dict_dir', type=str, default="Model/runs_html/chars_dict.p", help="Directory of word dictionary")
    parser.add_argument('--data_char_dict_dir', type=str, default="Model/runs_html/words_dict.p", help="Directory of character dictionary")

    # Model arguments
    parser.add_argument('--model_emb_dim', type=int, default=100, help="Embedding dimension size")
    parser.add_argument('--model_filter_sizes', type=str, default="3,4,5,6,7,8,9,10", help="Filter sizes of the convolution layer")

    # Test arguments
    parser.add_argument('--test_batch_size', type=int, default=20, help="Batch size")

    # Log arguments
    parser.add_argument('--log_output_dir', type=str, default="Model/runs_html", help="Output directory")
    parser.add_argument('--log_checkpoint_dir', type=str, default="Model/runs_html", help="Checkpoint directory")

    args = parser.parse_args()

    for key, val in vars(args).items():
        print(f"{key}={val}")

    # Prepare data
    content, labels_all = read_html(args.data_data_dir)

    # Remove entries where HTML content is None
    fails = []
    htmls = []
    labels = []
    for i in range(len(content)):
        if content[i] == None:
            fails.append(1)
        else:
            fails.append(0)
            htmls.append(content[i])
            labels.append(labels_all[i])

    print("Preprocessing HTMLS...")
    x, word_reverse_dict = get_word_vocab(htmls, args.data_max_len_words)
    word_x = get_words(x, word_reverse_dict, 1, htmls)

    # Load dictionaries
    ngram_dict = pickle.load(open(args.data_subword_dict_dir, "rb"))
    print("Size of subword vocabulary (train): {}".format(len(ngram_dict)))
    word_dict = pickle.load(open(args.data_word_dict_dir, "rb"))
    print("Size of word vocabulary (train): {}".format(len(word_dict)))
    ngrams_dict = ngram_dict
    chars_dict = pickle.load(open(args.data_char_dict_dir, "rb"))
    print("Size of character vocabulary (train): {}".format(len(chars_dict)))

    ngramed_id_x, worded_id_x = ngram_id_x_from_dict(word_x, args.data_max_len_subwords, ngram_dict, word_dict)
    chared_id_x = char_id_x(htmls, chars_dict, args.data_max_len_chars)

    print("Number of testing HTMLS: {}".format(len(labels)))

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
    cnn = HTMLNet(
        word_ngram_vocab_size=len(word_dict) + 1,
        char_vocab_size=len(chars_dict) + 1,
        embedding_size=args.model_emb_dim,
        word_seq_len=args.data_max_len_words,
        char_seq_len=args.data_max_len_chars,
        filter_sizes=list(map(int, args.model_filter_sizes.split(","))),
    )

    # Restore the checkpoint
    checkpoint = tf.train.Checkpoint(model=cnn)
    latest_checkpoint = tf.train.latest_checkpoint(args.log_checkpoint_dir)
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
        predictions = (logits.numpy() > 0.54).astype(int).flatten() # Adjust prediction threshold accordingly
        scores = logits.numpy().flatten()

        all_predictions.extend(predictions)
        all_scores.extend(scores)

    # Insert None entries back in (For consistency with other models)
    j = 0
    predictions = []
    scores = []
    for i in fails:
        if i == 0:
            predictions.append(all_predictions[j])
            scores.append(all_scores[j])
            j += 1
        else:
            predictions.append(None)
            scores.append(None)
                
    # Save test results
    output_file = os.path.join(args.log_output_dir, "test_results.txt")
    save_test_result(labels_all, predictions, scores, output_file)

if __name__ == "__main__":
    main()