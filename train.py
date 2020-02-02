import os
import random
import argparse
import sklearn

import numpy as np
import pickle as pkl
import csv
import torch
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from os.path import join
from keras.models import Input, Model
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers import GRU, CuDNNGRU, LSTM, CuDNNLSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from pytorch_pretrained_bert import BertTokenizer, BertModel
from keras.models import load_model
from data_generator import DataGenerator
from keras import backend as K
from lime.lime_text import LimeTextExplainer

epochs = 20
batch_size = 128
dropout_rate = 0.2
train = False
prop, nonprop = 0, 0

def bert_embeddings():
    data = list()
    with open('data/data.tsv', 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)

        for row in reader:
            if row[1] is "1" or (len(row[0]) > 50 and len(row[0]) < 200 and nonprop < 5000):
                data.append(row)
                if row[1] is "1":
                    prop = prop + 1
                elif row[1] is "0":
                    nonprop = nonprop + 1
    print("nonprop:" + str(nonprop))
    print("prop:" + str(prop))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    sentences_embeddings = dict()
    for row in tqdm(data):
        marked_text = '[CLS] ' + row[2] + ' [SEP]'
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)

        token_embeddings = []
        for token_i in range(len(tokenized_text)):
            hidden_layers = []
            for layer_i in range(len(encoded_layers)):
                vec = encoded_layers[layer_i][0][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)

        summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]

        sentences_embeddings[(row[0], row[1])] = (summed_last_4_layers, row[-1])

    with open('data-from-bert.pkl', 'wb') as file:
        pkl.dump(sentences_embeddings, file)

def recallAcc(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precisionAcc(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1Acc(y_true, y_pred):
    precision = precisionAcc(y_true, y_pred)
    recall = recallAcc(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

if train:
    #bert_embeddings()
    with open('data-from-bert-1.pkl', 'rb') as file:
        train_data = pkl.load(file)

    train_embeddings = list()
    train_labels = list()
    for example in train_data:
        if len(train_data[example][0]) == 2: continue
        train_embeddings.append([item.numpy() for item in train_data[example][0]])
        train_labels.append(int(train_data[example][1]))

    train_data = list(zip(train_embeddings, train_labels))
    random.shuffle(train_data)


    train_data = sorted(train_data, key=lambda item: len(item[0]))

    train_embeddings, train_labels = zip(*train_data)
    train_embeddings = np.array(train_embeddings)
    train_labels = np.array(train_labels)

    tokens_embeddings_input = Input(shape=(None, len(train_embeddings[0][0]),))

    lstm = Bidirectional(LSTM(units=128, dropout=dropout_rate, return_sequences=True,
                              kernel_initializer='he_normal'))(tokens_embeddings_input)

    lstm = Bidirectional(LSTM(units=128, dropout=dropout_rate, kernel_initializer='he_normal'))(lstm)

    dense = Dropout(dropout_rate)(Dense(units=256, activation='relu', kernel_initializer='he_normal')(lstm))
    dense = Dropout(dropout_rate)(Dense(units=128, activation='relu', kernel_initializer='he_normal')(dense))

    output = Dense(units=1, activation='sigmoid', kernel_initializer='he_normal')(dense)

    model = Model(tokens_embeddings_input, output)

    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy', f1Acc, precisionAcc, recallAcc])

    train_generator = DataGenerator(train_embeddings, train_labels, batch_size)

    checkpoint_cb = ModelCheckpoint(
        filepath='rnn-binary-bert-1.h5')

    model.fit_generator(
        generator=train_generator,
        epochs=epochs,
        callbacks=[checkpoint_cb]
    )

    model.save("model1.h5")

elif train is False:

    with open('data-from-bert-1.pkl', 'rb') as file:
        test_data = pkl.load(file)

    test_embeddings = list()
    test_labels = list()
    for example in test_data:
        if len(test_data[example][0]) == 2: continue
        test_embeddings.append([item.numpy() for item in test_data[example][0]])
        test_labels.append(int(test_data[example][1]))

    test_data = list(zip(test_embeddings, test_labels))
    random.shuffle(test_data)


    test_data = sorted(test_data, key=lambda item: len(item[0]))

    test_embeddings, test_labels = zip(*test_data)
    test_embeddings = np.array(test_embeddings)
    test_labels = np.array(test_labels)

    tokens_embeddings_input = Input(shape=(None, len(test_embeddings[0][0]),))
    test_generator = DataGenerator(test_embeddings, test_labels, batch_size)

    # load model
    model = load_model('model.h5')
    pred = model.predict_generator(test_generator)

