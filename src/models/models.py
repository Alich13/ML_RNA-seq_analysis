from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input

import os
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np 
import seaborn as sns
import importlib


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold ,cross_val_score

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from src import data
from src.utils.config import Config
from src.visualization import visualize
from src.data import make_dataset
from src.features import build_features
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model

def init_model():
    """

    Returns:
        model : the neural network initialized here below
    """
    init = 'random_uniform'
    input_layer = Input(shape=(100,))
    mid_layer = Dense(80, activation = 'relu', kernel_initializer = init)(input_layer)
    mid_layer_2 = Dense(30, activation = 'relu', kernel_initializer = init)(mid_layer)
    output_layer = Dense(5, activation = 'softmax', kernel_initializer = init)(mid_layer_2)
    DNN_model = Model(input_layer,output_layer)
    # sgd : Stochastic Gradient Descent
    DNN_model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
    DNN_model.summary()
    return DNN_model 

 
def encoder(X_train, X_test, factor ):
    """[summary]

    Args:
        X_train (np.array): the data we want to encode training set
        X_test (np.array): the data we want to encode testing set
        factor (int): the dimension division factor (the dimension of the original data will be divided by this factor )

    Returns:
        history : the history of the model training  
    """
   
    # number of input columns
    n_inputs = X_train.shape[1]  # same dimension as the original data
    # define encoder
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs*2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # bottleneck
    n_bottleneck = round(float(n_inputs) / 10)
    bottleneck = Dense(n_bottleneck)(e)
    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(n_inputs*2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # output layer
    output = Dense(n_inputs, activation='linear')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')
    # plot the autoencoder
    plot_model(model, Config.project_dir / 'reports/figures/autoencoder_no_compress.png', show_shapes=True)

    # fit the autoencoder model to reconstruct input
    history = model.fit(X_train, X_train, epochs=200, batch_size=16, verbose=2, validation_data=(X_test,X_test))

    # define an encoder model (without the decoder)
    encoder = Model(inputs=visible, outputs=bottleneck)
    plot_model(encoder, Config.project_dir / 'reports/figures/encoder_no_compress.png', show_shapes=True)
    # save the encoder to file
    encoder.save(Config.project_dir /'models/encoder.h5')

    return history

    