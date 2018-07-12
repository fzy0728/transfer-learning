# -*- coding: utf-8 -*-
"""
Created on  2018/5/18 13:30
@author: lhua
"""
import imp
import sys

imp.reload(sys)
import numpy as np
import pandas as pd
import jieba
import re
import random
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from gensim.corpora.dictionary import Dictionary
import multiprocessing

from keras.layers import BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import yaml
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Convolution1D, GlobalMaxPooling1D, MaxPooling1D, LSTM, GRU,Input, merge, Merge, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

np.random.seed(1337)  # For Reproducibility
# the dimension of word vector
vocab_dim = 64
# sentence length
maxlen = 100
# iter num
n_iterations = 1
# the number of words appearing
n_exposures = 10
# what is the maximum distance between the current word and the prediction word in a sentence, what is the maximum distance between the current and the prediction word in a sentence
window_size = 7
# batch size
batch_size = 32
# epoch num
n_epoch = 20
# input length
input_length = 100
# multi processing cpu number
cpu_count = multiprocessing.cpu_count()

#source
sourecefileneg = '../data/computerneg.xls'
sourecefilepos = '../data/computerpos.xls'

#target
targetfileneg = '../data/hotelneg.xls'
targetfilepos = '../data/hotelpos.xls'



# loading training file
def loadfile():
    sourceneg = pd.read_excel(sourecefileneg, header=None, index=None)
    sourcepos = pd.read_excel(sourecefilepos, header=None, index=None)
    targetneg = pd.read_excel(targetfileneg, header=None, index=None)
    targetpos = pd.read_excel(targetfilepos, header=None, index=None)
    #merge all data
    sourceneg = np.array(sourceneg[0])
    sourcepos = np.array(sourcepos[0])
    targetneg = np.array(targetneg[0])
    targetpos = np.array(targetpos[0])
    return sourceneg,sourcepos,targetneg,targetpos

#generating set of disused words
def getstopword(stopwordPath):
    stoplist = set()
    for line in stopwordPath:
        stoplist.add(line.strip())
        # print line.strip()
    return stoplist

#divide the sentence and remove the disused words
def wordsege(text):
    # get disused words set
    stopwordPath = open('../data/stopword.txt', 'r',encoding='utf-8')
    stoplist = getstopword(stopwordPath)
    stopwordPath.close()

    # divide the sentence and remove the disused words with jieba,return list
    text_list = []
    for document in text:

        seg_list = jieba.cut(document.strip())
        # seg_list = list(document.strip())
        fenci = []

        for item in seg_list:
            if item not in stoplist and re.match(r'-?\d+\.?\d*', item) == None and len(item.strip()) > 0:
                fenci.append(item)
        # if the word segmentation of the sentence is null,the label of the sentence should be deleted accordingly
        if len(fenci) > 0:
            text_list.append(fenci)
    return text_list

def tokenizer(neg, post,tarneg,tarpost):
    neg_sege = wordsege(neg)
    post_sege = wordsege(post)
    tarneg_sege = wordsege(tarneg)
    tarpost_sege = wordsege(tarpost)
    traincombined = np.concatenate((post_sege,neg_sege))
    textcombined = np.concatenate((tarpost_sege,tarneg_sege))
    # generating label and meging label data
    train_y = np.concatenate((np.ones(len(post_sege), dtype=int), np.zeros(len(neg_sege), dtype=int)))
    text_y = np.concatenate((np.ones(len(tarpost_sege), dtype=int), np.zeros(len(tarneg_sege), dtype=int)))
    return traincombined,train_y,textcombined,text_y


# create a dictionary of words and phrases,return the index of each word,vector of words,and index of words corresponding to each sentence
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        # the index of a word which have word vector is not 0
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        # integrate all the corresponding word vectors into the word vector matrix
        w2vec = {word: model[word] for word in w2indx.keys()}

        # a word without a word vector is indexed 0,return the index of word
        def parse_dataset(combined):
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in list(sentence):
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        # unify the length of the sentence with the pad_sequences function of keras
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        # return index, word vector matrix and the sentence with an unifying length and indexed
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


# the training of the word vector
def word2vec_train(combined):
    # model = Word2Vec(size=vocab_dim,
    #                  min_count=n_exposures,
    #                  window=window_size,
    #                  workers=cpu_count,
    #                  iter=n_iterations)
    # # build the vocabulary dictionary
    # model.build_vocab(combined)
    # # train the word vector model
    # model.train(combined, total_examples=model.corpus_count, epochs=50)
    # # save the trained model
    # model.save('data/Word2vec_model.pkl')
    # index, word vector matrix and the sentence with an unifying length and indexed based on the trained model
    word2model = Word2Vec.load('../model/Word2vec_new_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=word2model, combined=combined)

    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    # total number of word including the word without word vector
    n_symbols = len(index_dict) + 1
    # build word vector matrix which corresponding to the word index one by one
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    # partition test set and training set
    # x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    x_train = combined[:4000]
    x_test = combined[4000:]
    y_train = y[:4000]
    y_test = y[4000:]
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    # return the input parameters needed of the lstm model
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test

def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()

##定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    # model.add(Input(shape=(input_length,)))
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        # mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length,trainable=False))  # Adding Input Length
    model.add(Convolution1D(256, vocab_dim, border_mode='same', subsample_length=1,trainable=True))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print ('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    best_weights_filepath = '../model/best_lstm.h5'
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                    mode='auto')

    model.summary()
    print ("Train...")
    history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1,validation_data=(x_test, y_test),callbacks=[saveBestModel])

    print ("Evaluate...")
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)
    # save the trained lstm model
    yaml_string = model.to_yaml()
    with open('../model/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('../model/lstm.h5')
    print ('Test score:', score)
    print (training_vis(history))

# 训练模型，并保存
def train():
    print ('Loading Data...')
    sourceneg, sourcepos, targetneg, targetpos= loadfile()
    print('Tokenising...')
    traincombined, train_y, testcombined, text_y = tokenizer(sourceneg, sourcepos,targetneg, targetpos)
    print(len(traincombined), len(train_y),len(testcombined),len(text_y))
    sumcombined = np.concatenate((traincombined,testcombined))
    sumlabel = np.concatenate((train_y,text_y))
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(sumcombined)
    model = Word2Vec.load('../model/Word2vec_new_model.pkl')
    _, _, traincombined = create_dictionaries(model,traincombined)
    _, _, sumcombined = create_dictionaries(model,sumcombined)
    print(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    # n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, sumcombined, sumlabel)
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, sumcombined, sumlabel)
    print(y_test)
    print(x_train.shape, y_train.shape)
    # train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)
    train_DANN( x_train, y_train, x_test, y_test,enable_dann = True)

# building the input format data
def input_transform(string):
    words = list(jieba.cut(string))
    # reshape the list to bilayer list
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load('../model/Word2vec_model.pkl')
    # create a dictionary of words and phrases,return the index of each word,vector of words,and index of words corresponding to each senten
    _, _, combined = create_dictionaries(model, words)
    return combined


def lstm_predict(string):
    print('loading model......')
    with open('../model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('../model/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)
    print(data)
    # predict the new data
    result = model.predict_classes(data)
    if result[0][0] == 1:
        print(string, ' positive')
    else:
        print(string, ' negative')


def build_models(n_neurons):
    """Creates three different models, one used for source only training, two used for domain adaptation"""
    inputs = Input(shape=(100,))
    x3 = Convolution1D(128, vocab_dim, border_mode='same', subsample_length=1)(inputs)
    x3 = Activation("relu")(x3)
    x4 = Dense(n_neurons, activation='linear')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)

    source_classifier = Dense(2, activation='softmax', name="mo")(x4)
    domain_classifier = Dense(16, activation='linear', name="do4")(x4)
    domain_classifier = BatchNormalization(name="do5")(domain_classifier)
    domain_classifier = Activation("relu", name="do6")(domain_classifier)
    domain_classifier = Dropout(0.5)(domain_classifier)

    domain_classifier = Dense(2, activation='softmax', name="do")(domain_classifier)

    comb_model = Model(inputs=inputs, outputs=[source_classifier, domain_classifier])
    comb_model.compile(optimizer="Adam",
                       loss={'mo': 'categorical_crossentropy', 'do': 'categorical_crossentropy'},
                       loss_weights={'mo': 1, 'do': 2}, metrics=['accuracy'], )

    source_classification_model = Model(inputs=inputs, outputs=[source_classifier])
    source_classification_model.compile(optimizer="Adam",
                                        loss={'mo': 'categorical_crossentropy'}, metrics=['accuracy'], )

    domain_classification_model = Model(inputs=inputs, outputs=[domain_classifier])
    domain_classification_model.compile(optimizer="Adam",
                                        loss={'do': 'categorical_crossentropy'}, metrics=['accuracy'])

    embeddings_model = Model(inputs=inputs, outputs=[x4])
    embeddings_model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])

    return comb_model, source_classification_model, domain_classification_model, embeddings_model


def batch_generator(data, batch_size):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns batches of the same size
    This
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


def train_DANN(Xs, ys, Xt, yt, enable_dann=True, n_iterations=15000):
    # batch_size = 64

    model, source_classification_model, domain_classification_model, embeddings_model = build_models(64)

    y_class_dummy = np.ones((len(Xt), 2))
    y_adversarial_1 = to_categorical(np.array(([1] * batch_size + [0] * batch_size)))

    sample_weights_class = np.array(([1] * batch_size + [0] * batch_size))
    sample_weights_adversarial = np.ones((batch_size * 2,))

    S_batches = batch_generator([Xs, to_categorical(ys)], batch_size)
    T_batches = batch_generator([Xt, np.zeros(shape=(len(Xt), 2))], batch_size)

    for i in range(n_iterations):
        # print(y_class_dummy.shape, ys.shape)
        y_adversarial_2 = to_categorical(np.array(([0] * batch_size + [1] * batch_size)))

        X0, y0 = next(S_batches)
        X1, y1 = next(T_batches)

        X_adv = np.concatenate([X0, X1])
        y_class = np.concatenate([y0, np.zeros_like(y0)])

        adv_weights = []
        for layer in model.layers:
            if (layer.name.startswith("do")):
                adv_weights.append(layer.get_weights())

        if (enable_dann):
            # note - even though we save and append weights, the batchnorms moving means and variances
            # are not saved throught this mechanism
            stats = model.train_on_batch(X_adv, [y_class, y_adversarial_1],
                                         sample_weight=[sample_weights_class, sample_weights_adversarial])

            k = 0
            for layer in model.layers:
                if (layer.name.startswith("do")):
                    layer.set_weights(adv_weights[k])
                    k += 1

            class_weights = []

            for layer in model.layers:
                if (not layer.name.startswith("do")):
                    class_weights.append(layer.get_weights())

            stats2 = domain_classification_model.train_on_batch(X_adv, [y_adversarial_2])

            k = 0
            for layer in model.layers:
                if (not layer.name.startswith("do")):
                    layer.set_weights(class_weights[k])
                    k += 1

        else:
            source_classification_model.train_on_batch(X0, y0)

        if ((i + 1) % 1000 == 0):
            # print(i, stats)
            y_test_hat_t = source_classification_model.predict(Xt).argmax(1)
            y_test_hat_s = source_classification_model.predict(Xs).argmax(1)
            print("Iteration %d, source accuracy =  %.3f, target accuracy = %.3f" % (
            i, accuracy_score(ys, y_test_hat_s), accuracy_score(yt, y_test_hat_t)))
    return embeddings_model

def predict():
    sum = np.zeros(2000).reshape(2000,1)
    targetneg = pd.read_excel(targetfileneg, header=None, index=None)

    targetpos = pd.read_excel(targetfilepos, header=None, index=None)
    targetneg = np.random.permutation(np.array(targetneg[0]))
    targetpos = np.random.permutation(np.array(targetpos[0]))

    for  i in range(5):
        print('loading model......')
        with open('../model/lstm.yml', 'r') as f:
            yaml_string = yaml.load(f)
        model = model_from_yaml(yaml_string)
        print('loading weights......')
        model.load_weights('../model/best_lstm.h5')
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        word2model = Word2Vec.load('../model/Word2vec_new_model.pkl')

        for i in range(len(model.layers)):
            print(model.layers[i].name, i, model.layers[i].trainable)
        for i in range(4):
            model.layers[i].trainable = False
        sgd = SGD(lr=1e-4, decay=1e-4, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        targetneg_sege = wordsege(targetneg)
        targetpos_sege = wordsege(targetpos)
        texttarget_x = np.concatenate((targetneg_sege, targetpos_sege))
        texttarget_y = np.concatenate((np.zeros(len(targetneg_sege), dtype=int), np.ones(len(targetpos_sege), dtype=int)))
        # print(texttarget.reshape(-1,1).shape)
        # model.predict_classes(texttarget)
        texttarget_x.reshape(-1, 1)
        print(texttarget_x)
        _, _, combined = create_dictionaries(word2model, texttarget_x)
        print(combined)
        print('输出最好的模型结果：')
        print(model.evaluate(combined, texttarget_y))
        print('加入text进行微调：')
        # 训练抽取一半数据进行训练
        train_target_x = np.concatenate((combined[0:1000],combined[2000:3000]))
        train_target_y = np.concatenate((texttarget_y[0:1000],texttarget_y[2000:3000]))
        text_target_x =  np.concatenate((combined[1000:2000],combined[3000:]))
        text_target_y = np.concatenate((texttarget_y[1000:2000], texttarget_y[3000:]))
        print(text_target_y.shape)

        history = model.fit(train_target_x, train_target_y, batch_size=batch_size, nb_epoch=n_epoch, shuffle=True,
                  validation_data=(text_target_x, text_target_y), verbose=1)
        print('生成随机数')
        s = np.array(random.sample(list(combined),100))
        print (s)
        model.save_weights('../model/finll_lstm.h5')
        print(model.evaluate(text_target_x, text_target_y))
        print(model.predict(text_target_x))
        sum += model.predict(text_target_x)
        print(training_vis(history))
    result = ((1*((sum/5)>0.5)).T^text_target_y).sum()
    print(1-result/2000)


if __name__ == '__main__':
    train()
    # predict()

    # lstm_predict(string)