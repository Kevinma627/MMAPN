import numpy as np
import requests
import re
import tensorflow as tf
from bs4 import BeautifulSoup as bs
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

# Returns URLs and labels from a URL data file
def read_data(file_dir):
    with open(file_dir) as file:
        urls = []
        labels = []
        for line in file.readlines():
            items = line.strip().split('\t')
            label = int(items[0])
            labels.append(1 if label == 1 else 0)
            url = items[1]
            urls.append(url)
    return urls, labels

# ---------------------------------------------Word and Char Pre-Processing Functions---------------------------------------------
def get_word_vocab(urls, max_length_words, min_word_freq=0):
    tokenizer = Tokenizer(filters='', lower=False, oov_token='<UNKNOWN>')
    tokenizer.fit_on_texts(urls)
    word_counts = tokenizer.word_counts
    if min_word_freq > 1:
        low_freq_words = [word for word, count in word_counts.items() if count < min_word_freq]
        for word in low_freq_words:
            del tokenizer.word_index[word]
            del tokenizer.word_docs[word]
            del tokenizer.word_counts[word]
    x = tokenizer.texts_to_sequences(urls)
    x = pad_sequences(x, maxlen=max_length_words, padding='post', truncating='post')
    reverse_dict = {idx: word for word, idx in tokenizer.word_index.items()}
    return x, reverse_dict

def get_words(x, reverse_dict, delimit_mode, urls=None):
    processed_x = []
    if delimit_mode == 0:
        for url in x:
            words = []
            for word_id in url:
                if word_id != 0:
                    words.append(reverse_dict[word_id])
                else:
                    break
            processed_x.append(words)
    elif delimit_mode == 1:
        for i in range(len(x)):
            word_url = x[i]
            raw_url = urls[i]
            words = []
            idx = 0
            for word_id in word_url:
                if word_id == 0:
                    words.extend(list(raw_url[idx:]))
                    break
                else:
                    word = reverse_dict[word_id]
                    word_start_idx = raw_url.find(word, idx)
                    if word_start_idx == -1:
                        words.extend(list(raw_url[idx:]))
                        break
                    special_chars = list(raw_url[idx:word_start_idx])
                    words.extend(special_chars)
                    words.append(word)
                    idx = word_start_idx + len(word)
                    if idx >= len(raw_url):
                        break
            processed_x.append(words)
    return processed_x

def get_char_ngrams(ngram_len, word):
    word = "<" + word + ">"
    chars = list(word)
    ngrams = []
    for i in range(len(chars) - ngram_len + 1):
        ngram = ''.join(chars[i:i+ngram_len])
        ngrams.append(ngram)
    return ngrams

def char_id_x(urls, char_dict, max_len_chars):
    chared_id_x = []
    for url in urls:
        url_chars = list(url)[:max_len_chars]
        url_in_char_id = [char_dict.get(c, 0) for c in url_chars]
        chared_id_x.append(url_in_char_id)
    return chared_id_x

def ngram_id_x(word_x, max_len_subwords, high_freq_words=None):
    char_ngram_len = 1
    all_ngrams = set()
    ngramed_x = []
    all_words = set()
    worded_x = []
    high_freq_words_set = set(high_freq_words) if high_freq_words else None

    for url in word_x:
        url_in_ngrams = []
        url_in_words = []
        for word in url:
            ngrams = get_char_ngrams(char_ngram_len, word)
            if (len(ngrams) > max_len_subwords) or \
               (high_freq_words_set and len(word) > 1 and word not in high_freq_words_set):
                ngrams = ngrams[:max_len_subwords]
                all_ngrams.update(ngrams)
                url_in_ngrams.append(ngrams)
                all_words.add("<UNKNOWN>")
                url_in_words.append("<UNKNOWN>")
            else:
                all_ngrams.update(ngrams)
                url_in_ngrams.append(ngrams)
                all_words.add(word)
                url_in_words.append(word)
        ngramed_x.append(url_in_ngrams)
        worded_x.append(url_in_words)

    ngrams_dict = {ngram: idx+1 for idx, ngram in enumerate(sorted(all_ngrams))}
    print("Size of ngram vocabulary: {}".format(len(ngrams_dict)))
    words_dict = {word: idx+1 for idx, word in enumerate(sorted(all_words))}
    print("Size of word vocabulary: {}".format(len(words_dict)))
    print("Index of <UNKNOWN> word: {}".format(words_dict.get("<UNKNOWN>", "Not found")))

    ngramed_id_x = []
    for ngramed_url in ngramed_x:
        url_in_ngrams = []
        for ngramed_word in ngramed_url:
            ngram_ids = [ngrams_dict.get(ngram, 0) for ngram in ngramed_word]
            url_in_ngrams.append(ngram_ids)
        ngramed_id_x.append(url_in_ngrams)

    worded_id_x = []
    for worded_url in worded_x:
        word_ids = [words_dict.get(word, 0) for word in worded_url]
        worded_id_x.append(word_ids)

    return ngramed_id_x, ngrams_dict, worded_id_x, words_dict

def ngram_id_x_from_dict(word_x, max_len_subwords, ngram_dict, word_dict=None):
    char_ngram_len = 1
    ngramed_id_x = []
    worded_id_x = []
    word_vocab = set(word_dict.keys()) if word_dict else None

    for url in word_x:
        url_in_ngrams = []
        url_in_words = []
        for word in url:
            ngrams = get_char_ngrams(char_ngram_len, word)
            if len(ngrams) > max_len_subwords:
                ngrams = ngrams[:max_len_subwords]
                word = "<UNKNOWN>"

            ngrams_id = [ngram_dict.get(ngram, 0) for ngram in ngrams]
            url_in_ngrams.append(ngrams_id)

            if word_dict:
                word_id = word_dict.get(word, word_dict.get("<UNKNOWN>", 0))
            else:
                word_id = 0
            url_in_words.append(word_id)
        ngramed_id_x.append(url_in_ngrams)
        worded_id_x.append(url_in_words)

    return ngramed_id_x, worded_id_x

def get_ngramed_id_x(x_idxs, ngramed_id_x):
    return [ngramed_id_x[idx] for idx in x_idxs]
# --------------------------------------------------------------------------------------------------------------------------------

def save_test_result(labels, all_predictions, all_scores, output_dir):
    output_labels = [1 if i == 1 else -1 for i in labels]
    output_preds = []

    # Handles cases where prediction is None (For HTML model)
    for pred in all_predictions:
        if pred == None:
            output_preds.append(None)
        elif pred == 1:
            output_preds.append(1)
        else:
            output_preds.append(-1)

    # Calc accuracy ignoring Nones
    corr = 0
    count = 0
    for i in range(len(output_labels)):
        if(output_preds[i] != None):
            if(output_labels[i] == output_preds[i]):
                corr += 1
            count += 1

    accur = corr/count
    print("Test Accuracy: " + str(accur))

    with open(output_dir, "w") as file:
        file.write("label\tpredict\tscore\tAccuracy: " + str(accur) + "\n")
        for label, pred, score in zip(output_labels, output_preds, all_scores):
            file.write(f"{label}\t{pred}\t{score}\n")

# Fetch HTML content from a URL and uses BeautifulSoup to format into a string
def get_html(data):
    try:
        response = requests.get(data[0], timeout=3)  
        response.raise_for_status()  
        soup = bs(response.text, "html.parser")  
        html = soup.prettify()
        return (data[1], html)
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {data[0]}: {e}")
        return (data[1], None)

'''
- Read in html content and labels from a HTMl data file
- Reads in data in batches to avoid RAM issues
- Handles None case
'''
def read_html(file_path):
    htmls = []
    labels = []

    with open(file_path, "r") as file:
        buffer = []  
        label = None

        for line in file:
            match = re.match(r"^(\d+)\t\s*(<.*|None)", line)  
            if match:
                if buffer and label is not None:
                    if buffer == ["None"]:
                        htmls.append(None)
                    else:
                        htmls.append(clean_html("\n".join(buffer)))
                    
                    labels.append(label)

                # Start new entry
                label = int(match.group(1))
                buffer = [match.group(2)]
            else:
                buffer.append(line)

        # Process the last entry
        if buffer and label is not None:
            htmls.append(clean_html("\n".join(buffer)))
            labels.append(label)

    return htmls, labels

# Cleans common words and characters from HTML content that provide no valuable info to the model
def clean_html(html):
    if html.startswith("<!DOCTYPE html>"):
       html = html[15:]
    
    html = ''.join(c for c in html if c not in "<>/")
    words_to_remove = ["html", "head", "body", "title", "meta", "link", "style", "script", \
                        "div", "span", "p", "br", "hr", "header", "footer", "section", "article", "nav", "lang="]

    pattern = r'\b(?:' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
    html = re.sub(pattern, '', html)
    html = ' '.join(html.split())
    
    return html

'''
- Returns results from a specific sub model (takes in the test_results.txt file)
- Returns the max and min scores (or the max and min confidences in the case of TabNet)
- The isTab parameter is used to define if the model is TabNet 
    (If it is we must stop reading before we reach the feature importances at the bottom of the file)
- Handles the case when prediction and score is None (For HTMLNet)
'''
def get_results(file_path, isTab):
    labels = []
    preds = []
    scores = []

    min = 1
    max = 0

    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            items = line.strip().split('\t')

            if isTab and items[0] != '0' and items[0] != '1':
                break

            if items[1] != "None":
                items[2] = float(items[2])

                if items[2] < min:
                    min = items[2]
                if items[2] > max:
                    max = items[2]
                
                preds.append(int(items[1]))
            else:
                items[2] = None
                preds.append(None)

            labels.append(int(items[0]))
            scores.append(items[2])

    return np.column_stack((labels, preds, scores)), min, max

def make_batches(x_char_seq, x_train_word, y, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((x_char_seq, x_train_word, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    return dataset

def prep_batches(batch):
    x_batch_char_seq, x_batch_word, y_batch = batch
    x_batch_list = [x_batch_char_seq, x_batch_word]
    y_batch = tf.cast(y_batch, tf.float32)
    y_batch = tf.reshape(y_batch, (-1, 1))
    return x_batch_list, y_batch