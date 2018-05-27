

# -*- encoding:utf-8 -*-
import pandas as pd
# import tensorflow_hub as hub
import re
import numpy as np
import time
import h5py
from functools import reduce
from collections import Counter
import itertools 

import gensim
from gensim.models import Word2Vec

from keras.models import Sequential
from keras.models import Model  
from keras.models import load_model 
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import RepeatVector, Dense, Activation
# from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense,Flatten
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import Bidirectional

import jieba
import jieba.posseg as pseg

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA



def word_segmentation(sent):
    with open('./data/stopwords.txt', 'r') as f:
        stopwords = f.readlines()
    output = []
#     if isinstance(sent, unicode):
#         sent = sent
#     else:
#         sent = sent.decode('utf-8')
    words = pseg.cut(sent)
    for word, flag in words:
        if flag in ['n','a','v']:
            if word in stopwords:
                continue
            else:
                output.append(word)
    
    return output

#  One hot representation of sentences in encoder-decoder model output
def one_hot_encoder(sent, vocab_dict):
    max_len = max(map(len, sent))
    print('The max length of sentence: %s'%max_len)
    output = np.zeros((len(sent), max_len, len(vocab_dict)), dtype=np.bool)
    for s, words in enumerate(sent):
        for i, word in enumerate(words):
            output[s, i, vocab_dict[word]-1] = 1
            
    return output

#  Word2vector representation of sentence in encoder-decoder model input
def input_sequence(sent, word2vector):
    max_len = max(map(len, sent));
    # the vector length in word2vector is 64
    output = np.zeros((len(sent), max_len, 100), dtype=np.bool);
#     output = []
    for s, words in enumerate(sent):
#         sentence = []
        for i, word in enumerate(words):
            if word in word2vector:
                for j, v in enumerate(word2vector[word]):
                    output[s,i,j-1] = v
#                 sentence.append(word2vector[word])
#         output.append(sentence)
    return output

#  Encoder-decoder model, bidirectional LSTM as encoder, LSTM as decoder
def model_train(X, Y, input_dim, output_dim, hidden_dim):
    
    # input_maxlen = max(map(len,input_data_filter))

    clf = Sequential()
    # clf.add(Embedding(input_dim=64,output_dim=hidden_dim,input_length=input_maxlen))
    # encoder
    # clf.add(LSTM(hidden_dim, return_sequences=True))
    clf.add(Bidirectional(LSTM(hidden_dim,kernel_initializer='random_normal'), input_shape=input_dim))
    #decoder
    clf.add(RepeatVector(input_dim[0]))
    clf.add(LSTM(hidden_dim, return_sequences=True))
    # clf.add(Flatten())
    clf.add(TimeDistributed(Dense(output_dim)))
    # clf.add(Dense(hidden_dim,activation='relu'))
    clf.add(Activation('softmax'))
    print('Compiling...')
    time_start = time.time()
    clf.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    time_end = time.time()
    print('Compiled, cost time:%fsecond!' % (time_end - time_start))
    
    for iter_num in range(3):
        clf.fit(X, Y, batch_size=3, nb_epoch=1)
        
    return clf
    
#  Main function
def sentence_encode_LSTM(data, wordvector, dimension):
    
    data = np.array(data)
    print('word segmentation...')
    input_data = [word_segmentation(sent) for sent in data]
    
    vocab = reduce(lambda x, y: x | y, (set(lines) for lines in input_data))
    word_to_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    print('Perperaing input and output...')
    X_dim = [max(map(len,input_data)), wordvector.vector_size]
    Y_dim = len(vocab)
    
    X = input_sequence(input_data, wordvector)
    Y = one_hot_encoder(input_data, word_to_idx)

    print('Mode training...')
    model = model_train(X, Y, X_dim, Y_dim, dimension)
    LSTM_layer_model = Model(inputs=model.input,  
                                     outputs=model.layers[0].output) 
    sentence_encoder = LSTM_layer_model.predict(X)
    
    return sentence_encoder

def map_word_frequency(document):
    return Counter(itertools.chain(*document))

def sentence_IDF(tokenised_sentence_list, embedding_size, word_emb_model, PCA_orNot):

# Computing weighted average of the word vectors in the sentence;
# remove the projection of the average vectors on their first principal component.
# Borrowed from https://github.com/peter3125/sentence2vec; now compatible with python 2.7

    word_counts = map_word_frequency(tokenised_sentence_list)
    sentence_set=[]
    a = 1e-3
    for sentence in tokenised_sentence_list:
        vs = np.zeros(word_emb_model.vector_size)
        sentence_length = len(sentence)
        for word in sentence:
            a_value = a / (a + word_counts[word]) 
            if word in word_emb_model:
                vs = np.add(vs, np.multiply(a_value, word_emb_model[word])) # vs += sif * word_vector
            else: 
                vs = vs
        vs = np.divide(vs, sentence_length) # weighted average
        sentence_set.append(vs)
    
    sentence_set = np.array(sentence_set)
    where_are_nan = np.isnan(sentence_set) 
    sentence_set[where_are_nan] = 0

    # calculate PCA of this sentence set
    pca = PCA(n_components=embedding_size)
    pca.fit(sentence_set)
    u = pca.explained_variance_ratio_  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT
 
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below
 
    # resulting sentence vectors, vs = vs - u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u,vs)
        sentence_vecs.append(np.subtract(vs, sub))
    
    if PCA_orNot:
        return sentence_vecs
    else:
        return sentence_set

def sentence_encode_IDF(data, wordvector, dimension, pca_ornot = True):
    
    data = np.array(data)
    print('word segmentation...')
    input_data = [word_segmentation(sent) for sent in data]
    
#     vocab = reduce(lambda x, y: x | y, (set(lines) for lines in input_data))
#     word_to_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    print('sentence vector processing')
    sentence_encoder = sentence_IDF(input_data, dimension, wordvector, pca_ornot)
    
    return sentence_encoder

def sentence_keywords(tokenised_sentence_list, embedding_size, word_emb_model, PCA_orNot, weight, keywords):

    word_counts = map_word_frequency(tokenised_sentence_list)
    sentence_set=[]
    a = 1e-3
    for sentence in tokenised_sentence_list:
        vs = np.zeros(word_emb_model.vector_size)
        sentence_length = len(sentence)
        for word in sentence:
            if word in keywords:         
                if word in word_emb_model:
                    vs = np.add(vs, np.multiply(weight, word_emb_model[word])) # vs += sif * word_vector
                    sentence_length = sentence_length + weight - 1
                else: 
                    vs = vs
        vs = np.divide(vs, sentence_length) # weighted average
        sentence_set.append(vs)
    
    sentence_set = np.array(sentence_set)
    where_are_nan = np.isnan(sentence_set) 
    sentence_set[where_are_nan] = 0

    # calculate PCA of this sentence set
    pca = PCA(n_components=embedding_size)
    pca.fit(sentence_set)
    u = pca.explained_variance_ratio_  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT
 
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below
 
    # resulting sentence vectors, vs = vs - u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u,vs)
        sentence_vecs.append(np.subtract(vs, sub))
    
    if PCA_orNot:
        return sentence_vecs
    else:
        return sentence_set

def sentence_encode_keywords(data, wordvector, dimension, importance, keywords, pca_ornot = True):
    
    data = np.array(data)
    print('word segmentation...')
    input_data = [word_segmentation(sent) for sent in data]
    
#     vocab = reduce(lambda x, y: x | y, (set(lines) for lines in input_data))
#     word_to_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    print('sentence vector processing')
    sentence_encoder = sentence_keywords(input_data, dimension, wordvector, pca_ornot, importance, keywords)

    
    return sentence_encoder

