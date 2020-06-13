import itertools
from scipy import spatial
import os
import pickle
import string


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
