import pickle
import argparse
import json
import gc
import math
from util import *
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from keras.callbacks import ModelCheckpoint
from collections import defaultdict
from gensim.models import word2vec
from keras_han.model import HAN
from nltk.corpus import stopwords
import os


def main(dataset_path, print_flag=True):
    def train_word2vec(df, dataset_path):
        def get_embeddings(inp_data, vocabulary_inv, size_features=100,
                           mode='skipgram',
                           min_word_count=2,
                           context=5):
            num_workers = 15  # Number of threads to run in parallel
            downsampling = 1e-3  # Downsample setting for frequent words
            print('Training Word2Vec model...')
            sentences = [[vocabulary_inv[w] for w in s] for s in inp_data]
            if mode == 'skipgram':
                sg = 1
                print('Model: skip-gram')
            elif mode == 'cbow':
                sg = 0
                print('Model: CBOW')
            embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                                sg=sg,
                                                size=size_features,
                                                min_count=min_word_count,
                                                window=context,
                                                sample=downsampling)
            embedding_model.init_sims(replace=True)
            embedding_weights = np.zeros((len(vocabulary_inv) + 1, size_features))
            embedding_weights[0] = 0
            for i, word in vocabulary_inv.items():
                if word in embedding_model:
                    embedding_weights[i] = embedding_model[word]
                else:
                    embedding_weights[i] = np.random.uniform(-0.25, 0.25, embedding_model.vector_size)

            return embedding_weights

        tokenizer = fit_get_tokenizer(df.sentence, max_words=150000)
        print("Total number of words: ", len(tokenizer.word_index))
        tagged_data = tokenizer.texts_to_sequences(df.sentence)
        vocabulary_inv = {}
        for word in tokenizer.word_index:
            vocabulary_inv[tokenizer.word_index[word]] = word
        embedding_mat = get_embeddings(tagged_data, vocabulary_inv)
        pickle.dump(tokenizer, open(dataset_path + "tokenizer.pkl", "wb"))
        pickle.dump(embedding_mat, open(dataset_path + "embedding_matrix.pkl", "wb"))

    def preprocess(df, word_cluster):
        print("Preprocessing data..")
        stop_words = set(stopwords.words('english'))
        stop_words.add('would')
        word_vec = {}
        for index, row in df.iterrows():
            if index % 100 == 0:
                print("Finished rows: " + str(index) + " out of " + str(len(df)))
            line = row["sentence"]
            words = line.strip().split()
            new_words = []
            for word in words:
                try:
                    vec = word_vec[word]
                except:
                    vec = get_vec(word, word_cluster, stop_words)
                    if len(vec) == 0:
                        continue
                    word_vec[word] = vec
                new_words.append(word)
            df["sentence"][index] = " ".join(new_words)
        return df, word_vec

    def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
        def argmax_label(count_dict):
            maxi = 0
            max_label = None
            for l in count_dict:
                count = 0
                for t in count_dict[l]:
                    count += count_dict[l][t]
                if count > maxi:
                    maxi = count
                    max_label = l
            return max_label

        y = []
        X = []
        y_true = []
        index_word = {}
        for w in tokenizer.word_index:
            index_word[tokenizer.word_index[w]] = w
        for index, row in df.iterrows():
            line = row["sentence"]
            label = row["label"]
            tokens = tokenizer.texts_to_sequences([line])[0]
            words = []
            for tok in tokens:
                words.append(index_word[tok])
            count_dict = {}
            flag = 0
            for l in labels:
                seed_words = set()
                for w in label_term_dict[l]:
                    seed_words.add(w)
                int_labels = list(set(words).intersection(seed_words))
                if len(int_labels) == 0:
                    continue
                for word in words:
                    if word in int_labels:
                        flag = 1
                        try:
                            temp = count_dict[l]
                        except:
                            count_dict[l] = {}
                        try:
                            count_dict[l][word] += 1
                        except:
                            count_dict[l][word] = 1
            if flag:
                lbl = argmax_label(count_dict)
                if not lbl:
                    continue
                y.append(lbl)
                X.append(line)
                y_true.append(label)
        return X, y, y_true

    def train_classifier(df, labels, label_term_dict, label_to_index, index_to_label, dataset_path):
        print("Going to train classifier..")
        basepath = dataset_path
        model_name = "conwea"
        dump_dir = basepath + "models/" + model_name + "/"
        tmp_dir = basepath + "checkpoints/" + model_name + "/"
        os.makedirs(dump_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        max_sentence_length = 100
        max_sentences = 15
        max_words = 20000
        tokenizer = pickle.load(open(dataset_path + "tokenizer.pkl", "rb"))

        X, y, y_true = generate_pseudo_labels(df, labels, label_term_dict, tokenizer)
        y_one_hot = make_one_hot(y, label_to_index)
        print("Fitting tokenizer...")
        print("Splitting into train, dev...")
        X_train, y_train, X_val, y_val = create_train_dev(X, labels=y_one_hot, tokenizer=tokenizer,
                                                          max_sentences=max_sentences,
                                                          max_sentence_length=max_sentence_length,
                                                          max_words=max_words)
        print("Creating Embedding matrix...")
        embedding_matrix = pickle.load(open(dataset_path + "embedding_matrix.pkl", "rb"))
        print("Initializing model...")
        model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                    embedding_matrix=embedding_matrix)
        print("Compiling model...")
        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
        print("model fitting - Hierachical attention network...")
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        mc = ModelCheckpoint(filepath=tmp_dir + 'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', mode='max',
                             verbose=1, save_weights_only=True, save_best_only=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=100, batch_size=256, callbacks=[es, mc])
        print("****************** CLASSIFICATION REPORT FOR All DOCUMENTS ********************")
        X_all = prep_data(texts=df["sentence"], max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                          tokenizer=tokenizer)
        y_true_all = df["label"]
        pred = model.predict(X_all)
        pred_labels = get_from_one_hot(pred, index_to_label)
        print(classification_report(y_true_all, pred_labels))
        print("Dumping the model...")
        model.save_weights(dump_dir + "model_weights_" + model_name + ".h5")
        model.save(dump_dir + "model_" + model_name + ".h5")
        return pred_labels

    def expand_seeds(df, label_term_dict, pred_labels, label_to_index, index_to_label, word_to_index, index_to_word,
                     inv_docfreq, docfreq, it, n1, doc_freq_thresh=5):
        def get_rank_matrix(docfreq, inv_docfreq, label_count, label_docs_dict, label_to_index, term_count,
                            word_to_index, doc_freq_thresh):
            E_LT = np.zeros((label_count, term_count))
            components = {}
            for l in label_docs_dict:
                components[l] = {}
                docs = label_docs_dict[l]
                docfreq_local = calculate_doc_freq(docs)
                vect = CountVectorizer(vocabulary=list(word_to_index.keys()), tokenizer=lambda x: x.split())
                X = vect.fit_transform(docs)
                X_arr = X.toarray()
                rel_freq = np.sum(X_arr, axis=0) / len(docs)
                names = vect.get_feature_names()
                for i, name in enumerate(names):
                    try:
                        if docfreq_local[name] < doc_freq_thresh:
                            continue
                    except:
                        continue
                    E_LT[label_to_index[l]][word_to_index[name]] = (docfreq_local[name] / docfreq[name]) * inv_docfreq[
                        name] * np.tanh(rel_freq[i])
                    components[l][name] = {"reldocfreq": docfreq_local[name] / docfreq[name],
                                           "idf": inv_docfreq[name],
                                           "rel_freq": np.tanh(rel_freq[i]),
                                           "rank": E_LT[label_to_index[l]][word_to_index[name]]}
            return E_LT, components

        def disambiguate(label_term_dict, components):
            new_dic = {}
            for l in label_term_dict:
                all_interp_seeds = label_term_dict[l]
                seed_to_all_interp = {}
                disambiguated_seed_list = []
                for word in all_interp_seeds:
                    temp = word.split("$")
                    if len(temp) == 1:
                        disambiguated_seed_list.append(word)
                    else:
                        try:
                            seed_to_all_interp[temp[0]].add(word)
                        except:
                            seed_to_all_interp[temp[0]] = {word}

                for seed in seed_to_all_interp:
                    interpretations = seed_to_all_interp[seed]
                    max_interp = ""
                    maxi = -1
                    for interp in interpretations:
                        try:
                            if components[l][interp]["rank"] > maxi:
                                max_interp = interp
                                maxi = components[l][interp]["rank"]
                        except:
                            continue
                    disambiguated_seed_list.append(max_interp)
                new_dic[l] = disambiguated_seed_list
            return new_dic

        def expand(E_LT, index_to_label, index_to_word, it, label_count, n1, old_label_term_dict, label_docs_dict):
            word_map = {}
            zero_docs_labels = set()
            for l in range(label_count):
                if not np.any(E_LT):
                    continue
                elif len(label_docs_dict[index_to_label[l]]) == 0:
                    zero_docs_labels.add(index_to_label[l])
                else:
                    n = min(n1 * (it), int(math.log(len(label_docs_dict[index_to_label[l]]), 1.5)))
                    inds_popular = E_LT[l].argsort()[::-1][:n]
                    for word_ind in inds_popular:
                        word = index_to_word[word_ind]
                        try:
                            temp = word_map[word]
                            if E_LT[l][word_ind] > temp[1]:
                                word_map[word] = (index_to_label[l], E_LT[l][word_ind])
                        except:
                            word_map[word] = (index_to_label[l], E_LT[l][word_ind])

            new_label_term_dict = defaultdict(set)
            for word in word_map:
                label, val = word_map[word]
                new_label_term_dict[label].add(word)
            for l in zero_docs_labels:
                new_label_term_dict[l] = old_label_term_dict[l]
            return new_label_term_dict

        label_count = len(label_to_index)
        term_count = len(word_to_index)
        label_docs_dict = get_label_docs_dict(df, label_term_dict, pred_labels)

        E_LT, components = get_rank_matrix(docfreq, inv_docfreq, label_count, label_docs_dict, label_to_index,
                                           term_count, word_to_index, doc_freq_thresh)

        if it == 0:
            print("Disambiguating seeds..")
            label_term_dict = disambiguate(label_term_dict, components)
        else:
            print("Expanding seeds..")
            label_term_dict = expand(E_LT, index_to_label, index_to_word, it, label_count, n1, label_term_dict,
                                     label_docs_dict)
        return label_term_dict, components

    pkl_dump_dir = dataset_path
    df = pickle.load(open(pkl_dump_dir + "df_contextualized.pkl", "rb"))
    word_cluster = pickle.load(open(pkl_dump_dir + "word_cluster_map.pkl", "rb"))
    with open(pkl_dump_dir + "seedwords.json") as fp:
        label_term_dict = json.load(fp)

    label_term_dict = add_all_interpretations(label_term_dict, word_cluster)
    print_label_term_dict(label_term_dict, None, print_components=False)
    labels = list(set(label_term_dict.keys()))
    label_to_index, index_to_label = create_label_index_maps(labels)
    df, word_vec = preprocess(df, word_cluster)
    del word_cluster
    gc.collect()
    word_to_index, index_to_word = create_word_index_maps(word_vec)
    docfreq = calculate_df_doc_freq(df)
    inv_docfreq = calculate_inv_doc_freq(df, docfreq)

    train_word2vec(df, dataset_path)
    for i in range(5):
        print("ITERATION: ", i)
        pred_labels = train_classifier(df, labels, label_term_dict, label_to_index, index_to_label, dataset_path)
        label_term_dict, components = expand_seeds(df, label_term_dict, pred_labels, label_to_index, index_to_label,
                                                   word_to_index, index_to_word, inv_docfreq, docfreq, i, n1=5)
        if print_flag:
            print_label_term_dict(label_term_dict, components)
        print("#" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/nyt/')
    parser.add_argument('--gpu_id', type=str, default="cpu")
    args = parser.parse_args()
    if args.gpu_id != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(dataset_path=args.dataset_path)
