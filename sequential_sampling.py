import os
import GPyOpt
import numpy as np
from keras.layers import Activation, Dropout, BatchNormalization, Dense, Flatten
from keras.models import Sequential
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
import pickle
#%%
""" to import QMC- sobol samples """
import sobol_seq    ## Sobol

""" to import LHS : Latin hypercube samples """
from pyDOE import lhs ## LHS


#%%
def bayesian_opt(N,idx):
    """ Load the dataset """
#    x_train = pickle.load(open( 'Xtrain.pickle', 'rb'))
#    x_test = pickle.load(open( 'Xtest.pickle', 'rb'))
#    y_test = pickle.load(open( 'Ytest.pickle', 'rb'))
#    y_train = pickle.load(open( 'Ytrain.pickle', 'rb'))
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    """ Normalize the dataset """
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)
    
    num_classes = 10
    y_train = np_utils.to_categorical(y_train,num_classes)
    y_test = np_utils.to_categorical(y_test,num_classes)
        
    """ Build the CNN model """
    def CNN_model(lr,l1_drop,l2_drop,l3_drop,momentum):
        
        weight_decay = 10e-6
        model = Sequential()
        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(l1_drop))
        
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(l2_drop))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(l3_drop))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        

        sgd = optimizers.SGD(lr=lr, momentum = momentum)
        batch_size = 64
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='loss', min_delta=0.1, patience=3, verbose=0, mode='auto')
        
        model.fit(x_train, y_train, batch_size=batch_size, epochs=50,verbose=0,validation_data=(x_test,y_test),callbacks = [early_stop])    

        scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
        print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
        
        return scores
    
#%%
    """ Select the search space """
    
    bounds = [{'name': 'lr', 'type': 'continuous',  'domain': (0.0, 1.0)},
              {'name': 'l1_drop', 'type': 'continuous',  'domain': (0.0, 1.0)},
              {'name': 'l2_drop', 'type': 'continuous',  'domain': (0.0, 1.0)}, 
              {'name': 'l3_drop', 'type': 'continuous',  'domain': (0.0, 1.0)},              
              {'name': 'momentum', 'type': 'continuous',  'domain': (0.0, 1.0)}]
    
    
#%%
    t_acc=[]
    """ function to optimize CNN model """
    def f(x):
        evaluation = CNN_model(lr = float(x[:,0]), 
                               l1_drop = float(x[:,1]),
                               l2_drop = float(x[:,2]),
                               l3_drop = float(x[:,3]),
                               momentum = float(x[:,4]))
        
        print('samples : '+str(x))
        
        t_acc.append(evaluation[1])
        return evaluation[0]
    
    #%%
    feasible_region = GPyOpt.Design_space(space = bounds, constraints = None)
    
    """ Load samples of proposed design """
    script_path = os.getcwd()
    file_path = script_path +'/Spectral_samples'
    os.chdir(file_path)
    fileopen = 'stairpointset_'+str(N)+'_5_'+str(idx)+'.npy'
    initial_design =np.load(fileopen)    
    os.chdir(script_path)

    """ snippet for LHS design """
#    sample_design = lhs(dimension, samples = N)
    
    """ snippet for QMC (Sobol) design """
#    sample_design = sobol_seq.i4_sobol_generate(dimension,N)

    
    """ change scale of search space """
    def scale(A,B,x):
        y = x*x   
        return (1-y)*A + y*B
    
    initial_design[:,0] = scale(0.0001,0.5,initial_design[:,0])  ## learning rate
    initial_design[:,1] = scale(0.0,0.99,initial_design[:,1])  ## drop out 1
    initial_design[:,2] = scale(0.0,0.99,initial_design[:,2])  ## drop out 2
    initial_design[:,3] = scale(0.0,0.99,initial_design[:,3])  ## drop out 3
    initial_design[:,4] = scale(0.4,1.0,initial_design[:,4]) ## momentum
    
    #%%
    """ select objective """
    objective = GPyOpt.core.task.SingleObjective(f)
    
    """ select the model type """
    model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=2,verbose=False)
    
    """ the acquisition optimizer """
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
    
    """select the type of acquisition """
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)
    
    """ sequential sampling """
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    #%%
    """ Bayesian optimization """
    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)
    #%%
    
    max_time  = None 
    tolerance = 1e-8     # distance between two consecutive observations  
    iterations  = 100 - N
    for i in range(1,iterations):
        bo.run_optimization(max_iter = 1, max_time = max_time, eps = tolerance, verbosity=False) 
        print('Sample point = '+str(i))
        
    #%%
    t_acc=np.array(t_acc)
    t_acc = np.reshape(t_acc,(t_acc.shape[0],1))
    #%%
    sav_path = os.getcwd()
    name_file = '/bayesian_opt_results'
    if not os.path.exists(sav_path+name_file):
        os.mkdir(sav_path+name_file)
        
    os.chdir(sav_path+name_file)
    filesave = 'proposed_'+str(N)+'_'+str(idx)+'.npy'
    np.save(filesave,t_acc)
    os.chdir(sav_path)
    
    return t_acc

#%%

