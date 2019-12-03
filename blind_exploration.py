import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import numpy as np
import os

#%%
""" to import QMC- sobol samples """
import sobol_seq    ## Sobol

""" to import LHS : Latin hypercube samples """
from pyDOE import lhs ## LHS


#%%

def mnist_hypopt( N, idxx_num, data_per_class):
    num_classes = 10        
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
#%%
    """ prepare training dataset by sampling equal number of examples form each class  """    
    idx_classes = []
    for i in range(num_classes):
        idx_classes.append(np.where(y_train==i))
    
    traindata = []
    trainlabel = []
    for i in range(num_classes):
        idx = idx_classes[i]
        idx = idx[0][:data_per_class]
        traindata.append(x_train[idx,:])
        trainlabel.append(y_train[idx])

    train_data = np.array(traindata)
    train_data = train_data.reshape((data_per_class*num_classes,28,28))
    train_label = np.array(trainlabel)
    train_label = train_label.reshape((data_per_class*num_classes,))
    
    rand_idx = np.random.permutation(data_per_class*num_classes)
    
    x_train,y_train = train_data[rand_idx,:,:],train_label[rand_idx]
    
         
#%%
    
    """ Preprocess to feed into convolutional layers """
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
#%%    
    """ Blind exploration code """    
    
    def run_blindexploration(hyperparameters):
        
        lr=hyperparameters[0]
        dr1=hyperparameters[1]
        dr2=hyperparameters[2]
        dr3=hyperparameters[3]
        mo=hyperparameters[4]
        
        batch_size = 128
        epochs = 40
        """ Build a CNN model """
        model = Sequential()
        model.add(Conv2D(8, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dr1))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dr2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dr3))
        model.add(Dense(num_classes, activation='softmax'))
        sgd = SGD(lr=lr, decay=1e-8, momentum=mo, nesterov=True)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        early_stopping = EarlyStopping(patience=0.01, verbose=1)
        
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(x_test, y_test),
                  callbacks=[early_stopping])
        
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test accuracy:', score[1])
        
        test_acc = score[1]
        
        return test_acc
    #%%
    scale=[2,1,1,1,2]
    dimension = len(scale)
    
    """ snippet for proposed coverage based designs (5-D)"""
    script_path = os.getcwd()
    file_path = script_path +'/Spectral_samples'   
    os.chdir(file_path)    
    idx = idxx_num
    fileopen = 'stairpointset_'+str(N)+'_5_'+str(idx)+'.npy'
    sample_design = np.load(fileopen)
    os.chdir(script_path)
    
    """ snippet for LHS design """
#    sample_design = lhs(dimension, samples = N)
    
    """ snippet for QMC (Sobol) design """
#    sample_design = sobol_seq.i4_sobol_generate(dimension,N)

    
    """ Scale the search space """
    def scale_points(a,scale):
        dim=a.shape[1]
        for i in range(0,dim):
            a[:,i]=scale[i]*a[:,i]
            
        return a
    
    scaled_sample_design = scale_points(sample_design,scale)

    """ Start exploration process """

    def start_exploration(a): 
        test_acc=[]
        n=a.shape[0]
        
        for i in range(0,n):
            accuracy = run_blindexploration(a[i,:])
            test_acc.append(accuracy)   
        
        return test_acc
    #%%
    
    test_accuracy = start_exploration(scaled_sample_design)
    
    sav_path = os.getcwd()
    name_file = '/Spec_sfsd_5d'
    if not os.path.exists(sav_path+name_file):
        os.mkdir(sav_path+name_file)
        
    os.chdir(sav_path+name_file)
    filesave = 'spec'+str(N)+'_'+str(data_per_class)+'_'+str(idxx_num)+'.npy'
    np.save(filesave,test_accuracy)
    os.chdir(sav_path)
    
    return test_accuracy

#%%
