import pandas as pd
import numpy as np
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

	
    test_size = 0.1
    train_size = int(len(filtered_seqs) * (1 - test_size))
    X_train = filtered_seqs[:train_size]
    y_train = [dict[x] for x in X_train]
    X_test = filtered_seqs[train_size:]
    y_test = [dict[x] for x in X_test]

    return X_train,y_train,X_test,y_test

def twelve_bit_encode(X_train,y_train,X_test,y_test):

    #allocate ngrams
    characters = ['A', 'C', 'G', 'T']

    # integer encode each character
    label_encoder = LabelEncoder()
    integer_encoded_characters = label_encoder.fit_transform(characters)

    #One hot encode each character
    encoded_characters = to_categorical(integer_encoded_characters,4)
    encoded_characters = encoded_characters.astype(int)

    encoded_dict_ngrams = {}
    for i in range(len(characters)):
        encoded_dict_ngrams[characters[i]] = encoded_characters[i]

    X_train_encoded = []
    for dna_seq in X_train:
        f_start = 0
        f_step = 3
        s_start = 1
        s_step = 4
        list = []
        while (len(dna_seq) + 1 > s_step):
            ngram_one = dna_seq[f_start:f_step]
            vector_one = []
            for c in ngram_one:
                vector_one += encoded_dict_ngrams[c].tolist()

            ngram_two = dna_seq[s_start:s_step]
            vector_two = []
            for c in ngram_two:
                vector_two += encoded_dict_ngrams[c].tolist()

            #concatenate the vectors making it a 24bit vector
            conc_vector = vector_one + vector_two

            list.append(conc_vector)
            f_start += 1
            s_start += 1
            f_step += 1
            s_step += 1
        X_train_encoded.append(list)

    X_test_encoded = []
    for dna_seq in X_test:
        f_start = 0
        f_step = 3
        s_start = 1
        s_step = 4
        list = []
        while (len(dna_seq) + 1 > s_step):
            ngram_one = dna_seq[f_start:f_step]
            vector_one = []
            for c in ngram_one:
                vector_one += encoded_dict_ngrams[c].tolist()

            ngram_two = dna_seq[s_start:s_step]
            vector_two = []
            for c in ngram_two:
                vector_two += encoded_dict_ngrams[c].tolist()

            # concatenate the vectors making it a 24bit vector
            conc_vector = vector_one + vector_two

            list.append(conc_vector)
            f_start += 1
            s_start += 1
            f_step += 1
            s_step += 1
        X_test_encoded.append(list)

    # converting everything to ndarrays
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    X_train_encoded = np.asarray(X_train_encoded)
    X_test_encoded = np.asarray(X_test_encoded)

    return X_train_encoded,y_train,X_test_encoded,y_test