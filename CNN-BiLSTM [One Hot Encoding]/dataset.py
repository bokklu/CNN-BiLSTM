import pandas as pd
import numpy as np
import random
from itertools import product
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

def load_data():

    df = pd.read_csv(r'E:\School\University\DataScienceThesis\Histone Protein Datasets\H3Dataset.csv')
    dict = {}
    seqs = df['DNA Sequence'].values
    filtered_seqs = []
    for seq in seqs:
        if len(seq) == 500:
            filtered_seqs.append(seq)

    # Acquire data from csv into appropriate class list
    for i in range(len(df)):
        class_num = df.loc[i]['DNA Class']
        dna_seq = df.loc[i]['DNA Sequence']

        if len(dna_seq) == 500:
            dict[dna_seq] = class_num


    train_size = int(len(filtered_seqs) * (1 - 0.1))
    X_train = filtered_seqs[:train_size]
    y_train = [dict[x] for x in X_train]
    X_test = filtered_seqs[train_size:]
    y_test = [dict[x] for x in X_test]

    return X_train,y_train,X_test,y_test

def one_hot_encode_128(X_train,y_train,X_test,y_test):

    #allocate ngrams
    characters = ['A', 'C', 'G', 'T']
    ngrams = [''.join(i) for i in product(characters, repeat=3)]

    #integer encode each ngram
    label_encoder = LabelEncoder()
    integer_encoded_ngrams = label_encoder.fit_transform(ngrams)

    encoded_dict_ngrams = {}
    for i in range(len(ngrams)):
        encoded_dict_ngrams[ngrams[i]] = integer_encoded_ngrams[i]

    #create one hot encoded dict
    one_hot_encode_dict = {}
    one_hot_encoded = to_categorical(integer_encoded_ngrams)
    one_hot_encoded_int = one_hot_encoded.astype(int)
    count = 0
    for integer in integer_encoded_ngrams:
        one_hot_encode_dict[integer] = one_hot_encoded_int[count]
        count+=1

    X_train_one_hot_encoded = []
    for dna_seq in X_train:
        f_start = 0
        s_start = 1
        f_step = 3
        s_step = 4
        list = []
        while (len(dna_seq) + 1 > s_step):
            ngram_one = dna_seq[f_start:f_step]
            ngram_two = dna_seq[s_start:s_step]
            encoded_ngram_one = encoded_dict_ngrams[ngram_one]
            encoded_ngram_two = encoded_dict_ngrams[ngram_two]
            one_hot_vector_1 = one_hot_encode_dict[encoded_ngram_one]
            one_hot_vector_2 = one_hot_encode_dict[encoded_ngram_two]
            conc_array = np.concatenate((one_hot_vector_1,one_hot_vector_2),axis=0)
            list.append(conc_array.tolist())
            f_start+=1
            s_start+=1
            f_step+=1
            s_step+=1
        X_train_one_hot_encoded.append(list)

    X_test_one_hot_encoded = []
    for dna_seq in X_test:
        f_start = 0
        s_start = 1
        f_step = 3
        s_step = 4
        list = []
        while (len(dna_seq) + 1 > s_step):
            ngram_one = dna_seq[f_start:f_step]
            ngram_two = dna_seq[s_start:s_step]
            encoded_ngram_one = encoded_dict_ngrams[ngram_one]
            encoded_ngram_two = encoded_dict_ngrams[ngram_two]
            one_hot_vector_1 = one_hot_encode_dict[encoded_ngram_one]
            one_hot_vector_2 = one_hot_encode_dict[encoded_ngram_two]
            conc_array = np.concatenate((one_hot_vector_1, one_hot_vector_2), axis=0)
            list.append(conc_array.tolist())
            f_start += 1
            s_start += 1
            f_step += 1
            s_step += 1
        X_test_one_hot_encoded.append(list)

    #one hot encoding the output labels
    labels = [0,1]
    labels_one_hot_encoded = to_categorical(labels)
    labels_one_hot_encoded_int = labels_one_hot_encoded.astype(int)
    labels_one_hot_encoded_dict = {}
    count=0
    for n in labels:
        labels_one_hot_encoded_dict[n] = labels_one_hot_encoded_int[count].tolist()
        count+=1

    #formulate y_train
    y_train_encoded = []
    for i in y_train:
        y_train_encoded.append(labels_one_hot_encoded_dict[i])

    # formulate y_test
    y_test_encoded = []
    for i in y_test:
        y_test_encoded.append(labels_one_hot_encoded_dict[i])

    #converting everything to ndarrays
    y_train_encoded = np.asarray(y_train_encoded)
    y_test_encoded = np.asarray(y_test_encoded)
    X_train_encoded = np.asarray(X_train_one_hot_encoded)
    X_test_encoded = np.asarray(X_test_one_hot_encoded)

    return X_train_encoded,y_train_encoded,X_test_encoded,y_test_encoded