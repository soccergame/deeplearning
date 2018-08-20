#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:08:26 2017

@author: wufangyu
"""
import random
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras import losses
from load_data import get_siamese_data
from keras.layers import LeakyReLU

np.random.seed(1337)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred_label = []
    right_label = 0.0
    for item in predictions:
        if item < 0.5:
            pred_label.append(1)
        else:
            pred_label.append(0)
    for item in range(len(labels)):
        if pred_label[item] == labels[item]:
            right_label += 1.0

    return round(right_label / len(labels), 2)

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='tanh', padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', input_shape=tuple(input_dim)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='tanh', padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='tanh', padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.1))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.1))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))

    return model

def train_and_save_model(model_config):
    (pairs_train, labels_train), (pairs_test, labels_test) = get_siamese_data(model_config)

    index = [i for i in range(len(pairs_train))]
    random.shuffle(index)
    pairs_train = pairs_train[index]
    labels_train = labels_train[index]
    
    input_dim = tuple(model_config['input_dim'])
    batch_size = model_config['batch_size']
    epochs = model_config['epochs']
    validation_split = model_config['validation_split']
    
    base_network = create_base_network(input_dim)
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    
    model = Model(inputs=[input_a, input_b], outputs=distance)
    
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    
    model.fit([pairs_train[:, 0], pairs_train[:, 1]], labels_train,
              batch_size = batch_size,
              epochs = epochs,
              validation_split = validation_split)
    
    pred = model.predict([pairs_train[:, 0], pairs_train[:, 1]])
    tr_acc = compute_accuracy(pred, labels_train)
    pred = model.predict([pairs_test[:, 0], pairs_test[:, 1]])
    te_acc = compute_accuracy(pred, labels_test)
    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    model.save('siamese_cnn_cand.h5')


def load_model_with_contrastive_loss(model_name):
    losses.contrastive_loss = contrastive_loss
    return load_model(model_name)

