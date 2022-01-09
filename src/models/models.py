from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.svm import SVC
from tensorflow.keras.models import load_model
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
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from src import data
from src.utils.config import Config
from src.visualization import visualize
from src.data import make_dataset
from src.features import build_features
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model

encoder_path=Config.project_dir/ 'models/encoder.h5'



def KNN(X,Y,description:str):

    print (f"------------------KNN on {description}---------------------")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 10 ,shuffle=Y)
    model=KNeighborsClassifier()
    model.fit(X_train,Y_train)
    
    pred_test=model.predict(X_test)
    pred_train =model.predict(X_train)

    print(f"{description} training Accuracy = {accuracy_score(Y_train,pred_train)}")
    print(f"{description} test Accuracy = {accuracy_score(Y_test,pred_test)}")

    #cross validation
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)

    # report performance
    print('Cross validation Accuracy: %.3f std =(%.3f)' % (np.mean(scores), np.std(scores)))

   

    return model


def DT(X,Y,description:str, plot_ =False):

    print (f"------------------ Decision Tree on {description}---------------------")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 10 ,shuffle=Y)
    model=DecisionTreeClassifier()
    model.fit(X_train,Y_train)
    
    pred_test=model.predict(X_test)
    pred_train =model.predict(X_train)

    print(f"{description} training Accuracy = {accuracy_score(Y_train,pred_train)}")
    print(f"{description} test Accuracy = {accuracy_score(Y_test,pred_test)}")

    #cross validation
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)

    # report performance
    print('Cross validation Accuracy: %.3f std =(%.3f)' % (np.mean(scores), np.std(scores)))

    if plot_ ==True :

        plot_tree(model, filled=True)
        plt.title("Decision tree trained on all the features")
        plt.savefig(Config.project_dir /f"reports/figures/generated/tree_{description}.png")
        

    return model

def SVM(X,Y,description:str):

    print (f"------------------ SVM on {description}---------------------")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 10 ,shuffle=Y)
    model = SVC(kernel = 'linear', random_state = 10)
    model.fit(X_train, Y_train) 
    
    pred_test=model.predict(X_test)
    pred_train =model.predict(X_train)

    print(f"{description} training Accuracy = {accuracy_score(Y_train,pred_train)}")
    print(f"{description} test Accuracy = {accuracy_score(Y_test,pred_test)}")

    #cross validation
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)

    # report performance
    print('Cross validation Accuracy: %.3f std =(%.3f)' % (np.mean(scores), np.std(scores)))


    return model





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
    optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # compile autoencoder model
    model.compile(optimizer=optimizer, loss='mse')
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




def DNN (X ,labels_array:np.array,description:str ,encode_path=encoder_path ):
    """[summary]

    Args:
        X ([type]): [description]
        labels_array (np.array): [description]
        description (str): [description]
        encode_path ([type], optional): [description]. Defaults to encoder_path.
    """
    print (f"------------------ Neural Network on {description}---------------------")
    
    Y_encoded = []
    
    for i in labels_array :
        if i == 'PRAD' : Y_encoded.append(0)
        if i == 'LUAD': Y_encoded.append(1)
        if i == 'BRCA' : Y_encoded.append(2)
        if i == 'KIRC': Y_encoded.append(3)
        if i == 'COAD': Y_encoded.append(4)

    Y_bis = to_categorical(Y_encoded)
   
    X_train, X_test, y_train, y_test = train_test_split(X, Y_bis, test_size=0.33, random_state=42,stratify=labels_array)
    
    #define the model
    model = init_model()
    plot_model(model, to_file=Config.project_dir /f'reports/figures/generated/model_DNN_plot{description}.png', show_shapes=True, show_layer_names=True)
    

    # load the model from file
    encoder = load_model(Config.project_dir/ 'models/encoder.h5')
    # encode the train data
    X_train_encode = encoder.predict(X_train)
    # encode the test data
    X_test_encode = encoder.predict(X_test)

    # fit the model on the training set
    history=model.fit(X_train_encode,y_train,validation_split=0.33, batch_size=32, epochs=250, verbose=0)
    
   
    # plot loss
    plt.figure(figsize=(12,7))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')

    plt.title("loss = f(epoch)")
    plt.xlabel("epoch",fontsize=25)
    plt.ylabel("Loss",fontsize=25)
    plt.legend()
    plt.savefig(Config.project_dir /f'reports/figures/generated/DNN_loss{description}.png')

    #evaluation 

    Z_pred = model.predict(X_test_encode)
    prediction_test = np.argmax(Z_pred, axis = 1)
    y_test_not_bis_test = np.argmax(y_test, axis = 1) # test labels encoded [0,1,2,3,4] = [different cancer type]



    Z_train= model.predict(X_train_encode)
    prediction_training = np.argmax(Z_train, axis = 1)
    y_test_not_bis_training = np.argmax(y_train, axis = 1) # test labels encoded [0,1,2,3,4] = [different cancer type]

    print (f"-$$$$$$$$$$$$$$$ Neural Network on {description} ACCURACY----$$$$$$$$$$$$$$$--")
    print(f"{description} training Accuracy = {accuracy_score(y_test_not_bis_test,prediction_test)}")
    print(f"{description} test Accuracy = {accuracy_score(y_test_not_bis_training,prediction_training)}")
    print(pd.crosstab(y_test_not_bis_test,prediction_test))