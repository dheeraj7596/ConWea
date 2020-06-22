import itertools
from scipy import spatial
import os
import pickle
import string
import numpy as np
from nltk import tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def cosine_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)


def compute_pairwise_cosine_sim(tok_vecs):
    pairs = list(itertools.combinations(tok_vecs, 2))
    cos_sim = []
    for pair in pairs:
        sim = cosine_similarity(pair[0], pair[1])
        cos_sim.append(sim)
    return cos_sim


def read_bert_vectors(word, bert_dump_dir):
    word_clean = word.translate(str.maketrans('', '', string.punctuation))
    if os.path.isdir(os.path.join(bert_dump_dir, word_clean)):
        word_dir = os.path.join(bert_dump_dir, word_clean)
    elif os.path.isdir(os.path.join(bert_dump_dir, word)):
        word_dir = os.path.join(bert_dump_dir, word)
    else:
        raise Exception(word + " not found")
    filepaths = [os.path.join(word_dir, o) for o in os.listdir(word_dir) if
                 os.path.isfile(os.path.join(word_dir, o))]
    tok_vecs = []
    for path in filepaths:
        try:
            with open(path, "rb") as input_file:
                vec = pickle.load(input_file)
            tok_vecs.append(vec)
        except Exception as e:
            print("Exception while reading BERT pickle file: ", path, e)
    return tok_vecs


def get_relevant_dirs(bert_dump_dir):
    print("Getting relevant dirs..")
    dirs = os.listdir(bert_dump_dir)
    dir_dict = {}
    for dir in dirs:
        dir_dict[dir] = 1

    print("Dir dict ready..")
    dir_set = set()
    for i, dir in enumerate(dirs):
        if i % 1000 == 0:
            print("Finished checking dirs: " + str(i) + " out of: " + str(len(dirs)))
        dir_new = dir.translate(str.maketrans('', '', string.punctuation))
        if len(dir_new) == 0:
            continue
        try:
            temp = dir_dict[dir_new]
            dir_set.add(dir_new)
        except:
            dir_set.add(dir)
    return dir_set


def to_tokenized_string(sentence):
    tokenized = " ".join([t.text for t in sentence.tokens])
    return tokenized


def create_label_index_maps(labels):
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return label_to_index, index_to_label


def make_one_hot(y, label_to_index):
    labels = list(label_to_index.keys())
    n_classes = len(labels)
    y_new = []
    for label in y:
        current = np.zeros(n_classes)
        i = label_to_index[label]
        current[i] = 1.0
        y_new.append(current)
    y_new = np.asarray(y_new)
    return y_new


def prep_data(max_sentence_length, max_sentences, texts, tokenizer):
    data = np.zeros((len(texts), max_sentences, max_sentence_length), dtype='int32')
    documents = []
    for text in texts:
        sents = tokenize.sent_tokenize(text)
        documents.append(sents)
    for i, sentences in enumerate(documents):
        tokenized_sentences = tokenizer.texts_to_sequences(
            sentences
        )
        tokenized_sentences = pad_sequences(
            tokenized_sentences, maxlen=max_sentence_length
        )

        pad_size = max_sentences - tokenized_sentences.shape[0]

        if pad_size < 0:
            tokenized_sentences = tokenized_sentences[0:max_sentences]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0, pad_size), (0, 0)),
                mode='constant', constant_values=0
            )

        data[i] = tokenized_sentences[None, ...]
    return data


def create_train_dev(texts, labels, tokenizer, max_sentences=15, max_sentence_length=100, max_words=20000):
    data = prep_data(max_sentence_length, max_sentences, texts, tokenizer)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    return X_train, y_train, X_test, y_test


def get_from_one_hot(pred, index_to_label):
    pred_labels = np.argmax(pred, axis=-1)
    ans = []
    for l in pred_labels:
        ans.append(index_to_label[l])
    return ans


def calculate_df_doc_freq(df):
    docfreq = {}
    docfreq["UNK"] = len(df)
    for index, row in df.iterrows():
        line = row["sentence"]
        words = line.strip().split()
        temp_set = set(words)
        for w in temp_set:
            try:
                docfreq[w] += 1
            except:
                docfreq[w] = 1
    return docfreq


def calculate_doc_freq(docs):
    docfreq = {}
    for doc in docs:
        words = doc.strip().split()
        temp_set = set(words)
        for w in temp_set:
            try:
                docfreq[w] += 1
            except:
                docfreq[w] = 1
    return docfreq


def calculate_inv_doc_freq(df, docfreq):
    inv_docfreq = {}
    N = len(df)
    for word in docfreq:
        inv_docfreq[word] = np.log(N / docfreq[word])
    return inv_docfreq


def create_word_index_maps(word_vec):
    word_to_index = {}
    index_to_word = {}
    words = list(word_vec.keys())
    for i, word in enumerate(words):
        word_to_index[word] = i
        index_to_word[i] = word
    return word_to_index, index_to_word


def get_vec(word, word_cluster, stop_words):
    if word in stop_words:
        return []
    t = word.split("$")
    if len(t) == 1:
        prefix = t[0]
        cluster = 0
    elif len(t) == 2:
        prefix = t[0]
        cluster = t[1]
        try:
            cluster = int(cluster)
        except:
            prefix = word
            cluster = 0
    else:
        prefix = "".join(t[:-1])
        cluster = t[-1]
        try:
            cluster = int(cluster)
        except:
            cluster = 0

    word_clean = prefix.translate(str.maketrans('', '', string.punctuation))
    if len(word_clean) == 0 or word_clean in stop_words:
        return []
    try:
        vec = word_cluster[word_clean][cluster]
    except:
        try:
            vec = word_cluster[prefix][cluster]
        except:
            try:
                vec = word_cluster[word][0]
            except:
                vec = []
    return vec


def get_label_docs_dict(df, label_term_dict, pred_labels):
    label_docs_dict = {}
    for l in label_term_dict:
        label_docs_dict[l] = []
    for index, row in df.iterrows():
        line = row["sentence"]
        label_docs_dict[pred_labels[index]].append(line)
    return label_docs_dict


def add_all_interpretations(label_term_dict, word_cluster):
    print("Considering all interpretations of seed words..")
    new_dic = {}
    for l in label_term_dict:
        for word in label_term_dict[l]:
            try:
                cc = word_cluster[word]
                n_inter = len(cc)
            except:
                continue

            if n_inter == 1:
                try:
                    new_dic[l].append(word)
                except:
                    new_dic[l] = [word]
            else:
                for i in range(n_inter):
                    con_word = word + "$" + str(i)
                    try:
                        new_dic[l].append(con_word)
                    except:
                        new_dic[l] = [con_word]
    return new_dic


def print_label_term_dict(label_term_dict, components, print_components=True):
    for label in label_term_dict:
        print(label)
        print("*" * 80)
        for val in label_term_dict[label]:
            try:
                if print_components:
                    print(val, components[label][val])
                else:
                    print(val)
            except Exception as e:
                print("Exception occurred: ", e, val)


def fit_get_tokenizer(data, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    return tokenizer
