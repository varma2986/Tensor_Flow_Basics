import openpyxl
#import xlrd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import random
import math

np.random.seed(1)
tf.set_random_seed(1)
seed=1
adam_learning_rate = 0.0001
adam_epsilon = 1
#num_epochs = 10
num_epochs = 30000
accuracy_tolerance = 0.1

def get_accuracy(X, Y):
    params = json.load(open('params.json'))
    mispredictedindices = []

    W1 = eval(params["W1"])
    W2 = eval(params["W2"])
    W3 = eval(params["W3"])
    W4 = eval(params["W4"])
    W5 = eval(params["W5"])
    W6 = eval(params["W6"])
    W7 = eval(params["W7"])
    W8 = eval(params["W8"])
    W9 = eval(params["W9"])
    W10 = eval(params["W10"])
    W11 = eval(params["W11"])
    W12 = eval(params["W12"])

    W13 = eval(params["W13"])
    W14 = eval(params["W14"])
    W15 = eval(params["W15"])
    W16 = eval(params["W16"])
    W17 = eval(params["W17"])
    W18 = eval(params["W18"])
    W19 = eval(params["W19"])
    W20 = eval(params["W20"])
    W21 = eval(params["W21"])
    W22 = eval(params["W22"])
    W23 = eval(params["W23"])
    W24 = eval(params["W24"])
    W25 = eval(params["W25"])
    W26 = eval(params["W26"])
    W27 = eval(params["W27"])
    W28 = eval(params["W28"])
    W29 = eval(params["W29"])
    W30 = eval(params["W30"])
    W31 = eval(params["W31"])
    W32 = eval(params["W32"])
    W33 = eval(params["W33"])
    W34 = eval(params["W34"])
    W35 = eval(params["W35"])
    W36 = eval(params["W36"])
    W37 = eval(params["W37"])
    W38 = eval(params["W38"])
    W39 = eval(params["W39"])
    W40 = eval(params["W40"])
    W41 = eval(params["W41"])
    W42 = eval(params["W42"])
    W43 = eval(params["W43"])
    W44 = eval(params["W44"])
    W45 = eval(params["W45"])
    W46 = eval(params["W46"])
    W47 = eval(params["W47"])
    W48 = eval(params["W48"])
    W49 = eval(params["W49"])
    W50 = eval(params["W50"])
    W51 = eval(params["W51"])
    W52 = eval(params["W52"])
    W53 = eval(params["W53"])
    W54 = eval(params["W54"])
    W55 = eval(params["W55"])
    W56 = eval(params["W56"])
    W57 = eval(params["W57"])
    W58 = eval(params["W58"])
    W59 = eval(params["W59"])
    W60 = eval(params["W60"])
    W61 = eval(params["W61"])
    W62 = eval(params["W62"])
    W63 = eval(params["W63"])
    W64 = eval(params["W64"])
    W65 = eval(params["W65"])
    W66 = eval(params["W66"])
    W67 = eval(params["W67"])

    b = eval(params["b"])
    X_train_mu0 = eval(params["X_train_mu0"])
    X_train_mu1 = eval(params["X_train_mu1"])
    X_train_mu2 = eval(params["X_train_mu2"])
    X_train_mu3 = eval(params["X_train_mu3"])
    X_train_mu4 = eval(params["X_train_mu4"])
    X_train_mu5 = eval(params["X_train_mu5"])
    X_train_mu6 = eval(params["X_train_mu6"])
    X_train_mu7 = eval(params["X_train_mu7"])
    X_train_mu8 = eval(params["X_train_mu8"])

    X_train_sigma0 = eval(params["X_train_sigma0"])
    X_train_sigma1 = eval(params["X_train_sigma1"])
    X_train_sigma2 = eval(params["X_train_sigma2"])
    X_train_sigma3 = eval(params["X_train_sigma3"])
    X_train_sigma4 = eval(params["X_train_sigma4"])
    X_train_sigma5 = eval(params["X_train_sigma5"])
    X_train_sigma6 = eval(params["X_train_sigma6"])
    X_train_sigma7 = eval(params["X_train_sigma7"])
    X_train_sigma8 = eval(params["X_train_sigma8"])

    Y_train_mu = eval(params["Y_train_mu"])
    Y_train_sigma = eval(params["Y_train_sigma"])


    num_accurate_datasets = 0
    num_total_datasets = X.shape[0]
    accuracy = 0.0
    
    print("X_train_mu 0 is",X_train_mu0)
    X_norm = np.zeros((X.shape[0],X.shape[1]))

    X_norm[:,0] = (X[:,0] - X_train_mu0)/X_train_sigma0
    X_norm[:,1] = (X[:,1] - X_train_mu1)/X_train_sigma1
    X_norm[:,2] = (X[:,2] - X_train_mu2)/X_train_sigma2
    X_norm[:,3] = (X[:,3] - X_train_mu3)/X_train_sigma3
    X_norm[:,4] = (X[:,4] - X_train_mu4)/X_train_sigma4
    X_norm[:,5] = (X[:,5] - X_train_mu5)/X_train_sigma5
    X_norm[:,6] = (X[:,6] - X_train_mu6)/X_train_sigma6
    X_norm[:,7] = (X[:,7] - X_train_mu7)/X_train_sigma7
    X_norm[:,8] = (X[:,8] - X_train_mu8)/X_train_sigma8

    #for i in range(X.shape[1]):
    #    X_norm[:,i] = (X.shape[:,i] - X_train_mu[i])/X_train_sigma[i]

    for i in range(X_norm.shape[0]):

        Y9 = np.multiply(W62, np.multiply(np.multiply(X_norm[i,2], X_norm[i,8]), X_norm[i,0])) +  np.multiply(W63, np.multiply(np.multiply(X_norm[i,2], X_norm[i,8]), X_norm[i,1])) +  np.multiply(W64, np.multiply(np.multiply(X_norm[i,2], X_norm[i,8]), X_norm[i,3])) +  np.multiply(W65, np.multiply(np.multiply(X_norm[i,2], X_norm[i,8]), X_norm[i,4])) +  np.multiply(W66, np.multiply(np.multiply(X_norm[i,2], X_norm[i,8]), X_norm[i,5])) +  np.multiply(W67, np.multiply(np.multiply(X_norm[i,2], X_norm[i,8]), X_norm[i,6]))

        Y8 = np.multiply(W55, np.multiply(np.multiply(X_norm[i,7], X_norm[i,8]),X_norm[i,2])) + np.multiply(W56, np.multiply(np.multiply(X_norm[i,7], X_norm[i,8]),X_norm[i,3])) + np.multiply(W57, np.multiply(np.multiply(X_norm[i,7], X_norm[i,8]),X_norm[i,4])) + np.multiply(W58, np.multiply(np.multiply(X_norm[i,7], X_norm[i,8]),X_norm[i,5])) + np.multiply(W59, np.multiply(np.multiply(X_norm[i,7], X_norm[i,8]),X_norm[i,6])) + np.multiply(W60, np.multiply(np.multiply(X_norm[i,7], X_norm[i,8]),X_norm[i,0])) + np.multiply(W61, np.multiply(np.multiply(X_norm[i,7], X_norm[i,8]),X_norm[i,1])) 

        Y7 = np.multiply(W52, np.multiply(X_norm[i,1], X_norm[i,3])) + np.multiply(W53, np.multiply(X_norm[i,1], X_norm[i,8])) + np.multiply(W54, np.multiply(X_norm[i,3], X_norm[i,8]))

        Y6 = np.multiply(W49, np.multiply(X_norm[i,7], X_norm[i,1])) + np.multiply(W50, np.multiply(X_norm[i,7], X_norm[i,3])) + np.multiply(W51, np.multiply(X_norm[i,7], X_norm[i,8]))

        Y5 = np.multiply(W45, np.multiply(X_norm[i,6], X_norm[i,1])) + np.multiply(W46, np.multiply(X_norm[i,6], X_norm[i,3])) + np.multiply(W47, np.multiply(X_norm[i,6], X_norm[i,7])) + np.multiply(W48, np.multiply(X_norm[i,6], X_norm[i,8]))

        Y4 = np.multiply(W40, np.multiply(X_norm[i,4], X_norm[i,1])) + np.multiply(W41, np.multiply(X_norm[i,4], X_norm[i,3])) + np.multiply(W42, np.multiply(X_norm[i,4], X_norm[i,6])) + np.multiply(W43, np.multiply(X_norm[i,4], X_norm[i,7])) + np.multiply(W44, np.multiply(X_norm[i,4], X_norm[i,8]))

        Y3 = np.multiply(W34, np.multiply(X_norm[i,5], X_norm[i,1])) +  np.multiply(W35, np.multiply(X_norm[i,5], X_norm[i,3])) +  np.multiply(W36, np.multiply(X_norm[i,5], X_norm[i,4])) +  np.multiply(W37, np.multiply(X_norm[i,5], X_norm[i,6])) +  np.multiply(W38, np.multiply(X_norm[i,5], X_norm[i,7])) +  np.multiply(W39, np.multiply(X_norm[i,5], X_norm[i,8])) 

        Y2 = np.multiply(W31, np.multiply(X_norm[i,0], X_norm[i,0])) + np.multiply(W32, np.multiply(X_norm[i,1], X_norm[i,1])) + np.multiply(W33, np.multiply(X_norm[i,7],X_norm[i,8]))

        Y1 = np.multiply(W1,X_norm[i,0]) + np.multiply(W2,X_norm[i,1]) + np.multiply(W3,X_norm[i,2]) + np.multiply(W4, X_norm[i,3]) + np.multiply(W5, X_norm[i,4]) + np.multiply(W6, X_norm[i,5]) + np.multiply(W7, X_norm[i,6]) + np.multiply(W8, X_norm[i,7]) + np.multiply(W9, X_norm[i,8]) + np.multiply(W10, np.multiply(X_norm[i,2], X_norm[i,3])) + np.multiply(W11, np.multiply(X_norm[i,2], X_norm[i,4])) + np.multiply(W12, np.multiply(X_norm[i,2], X_norm[i,5])) + np.multiply(W13, np.multiply(X_norm[i,2], X_norm[i,6])) + np.multiply(W14, np.multiply(X_norm[i,2], X_norm[i,7])) + np.multiply(W15, np.multiply(X_norm[i,2], X_norm[i,8])) +  np.multiply(W16, np.multiply(X_norm[i,2], X_norm[i,2])) + np.multiply(W17, np.multiply(X_norm[i,3], X_norm[i,3])) + np.multiply(W18, np.multiply(X_norm[i,4], X_norm[i,4])) + np.multiply(W19, np.multiply(X_norm[i,5], X_norm[i,5])) + np.multiply(W20, np.multiply(X_norm[i,6], X_norm[i,6])) + np.multiply(W21, np.multiply(X_norm[i,7], X_norm[i,7])) + np.multiply(W22, np.multiply(X_norm[i,8], X_norm[i,8])) + np.multiply(W23, np.multiply(X_norm[i,0], X_norm[i,1])) + np.multiply(W24, np.multiply(X_norm[i,0], X_norm[i,2])) + np.multiply(W25, np.multiply(X_norm[i,0], X_norm[i,3])) + np.multiply(W26, np.multiply(X_norm[i,0], X_norm[i,4])) + np.multiply(W27, np.multiply(X_norm[i,0], X_norm[i,5])) + np.multiply(W28, np.multiply(X_norm[i,0], X_norm[i,6])) + np.multiply(W29, np.multiply(X_norm[i,0], X_norm[i,7])) + np.multiply(W30, np.multiply(X_norm[i,0], X_norm[i,8]))  + b

        Y_pred_norm = Y1 + Y2 + Y3 + Y4 + Y5 + Y6 + Y7 + Y8 + Y9
        #Y_pred_norm = np.multiply(W1,X_norm[i,0]) + np.multiply(W2,X_norm[i,1]) + np.multiply(W3,X_norm[i,2]) + np.multiply(W4, X_norm[i,3]) + np.multiply(W5, X_norm[i,4]) + np.multiply(W6, X_norm[i,5]) + np.multiply(W7, X_norm[i,6]) + np.multiply(W8, X_norm[i,7]) + np.multiply(W9, np.multiply(X_norm[i,0], X_norm[i,0])) + np.multiply(W10, np.multiply(X_norm[i,1], X_norm[i,1])) + np.multiply(W11, np.multiply(X_norm[i,2], X_norm[i,2])) + np.multiply(W12, np.multiply(X_norm[i,3], X_norm[i,3])) + np.multiply(W13, np.multiply(X_norm[i,4], X_norm[i,4])) + np.multiply(W14, np.multiply(X_norm[i,5], X_norm[i,5])) + np.multiply(W15, np.multiply(X_norm[i,6], X_norm[i,6])) + np.multiply(W16, np.multiply(X_norm[i,7], X_norm[i,7])) + np.multiply(W17, np.multiply(X_norm[i,2],X_norm[i,3])) + np.multiply(W18, np.multiply(X_norm[i,2],X_norm[i,4])) + np.multiply(W19, np.multiply(X_norm[i,2],X_norm[i,5])) + np.multiply(W20, np.multiply(X_norm[i,2],X_norm[i,6])) + np.multiply(W21, np.multiply(X_norm[i,2],X_norm[i,7])) + np.multiply(W22, np.multiply(X_norm[i,7],X_norm[i,3])) + np.multiply(W23, np.multiply(X_norm[i,7],X_norm[i,4])) + np.multiply(W24, np.multiply(X_norm[i,7],X_norm[i,5])) + np.multiply(W25, np.multiply(X_norm[i,7],X_norm[i,6])) + np.multiply(W26, np.multiply(X_norm[i,6], X_norm[i,5])) + np.multiply(W27, np.multiply(X_norm[i,6], X_norm[i,4])) + np.multiply(W28, np.multiply(X_norm[i,6], X_norm[i,3])) + np.multiply(W29, np.multiply(X_norm[i,5], X_norm[i,3])) + np.multiply(W30, np.multiply(X_norm[i,5], X_norm[i,4])) + np.multiply(W31, np.multiply(X_norm[i,5], X_norm[i,6])) + np.multiply(W32, np.multiply(X_norm[i,3], X_norm[i,4])) + np.multiply(W33, np.multiply(X_norm[i,0], X_norm[i,2])) +  np.multiply(W34, np.multiply(X_norm[i,0], X_norm[i,3])) +  np.multiply(W35, np.multiply(X_norm[i,0], X_norm[i,4])) +  np.multiply(W36, np.multiply(X_norm[i,0], X_norm[i,5])) +  np.multiply(W37, np.multiply(X_norm[i,0], X_norm[i,6])) +  np.multiply(W38, np.multiply(X_norm[i,0], X_norm[i,7])) + np.multiply(W39, np.multiply(X_norm[i,1], X_norm[i,0])) + np.multiply(W40, np.multiply(X_norm[i,1], X_norm[i,3])) + np.multiply(W41, np.multiply(X_norm[i,1], X_norm[i,4])) + np.multiply(W42, np.multiply(X_norm[i,1], X_norm[i,5])) + np.multiply(W43, np.multiply(X_norm[i,1], X_norm[i,6])) + np.multiply(W44, np.multiply(X_norm[i,1], X_norm[i,7])) + np.multiply(W45, np.multiply(X_norm[i,2], np.multiply(X_norm[i,4], X_norm[i,5]))) + np.multiply(W46, np.multiply(X_norm[i,2], np.multiply(X_norm[i,4], X_norm[i,6]))) + np.multiply(W47, np.multiply(X_norm[i,2], np.multiply(X_norm[i,4], X_norm[i,7]))) + np.multiply(W48, np.multiply(X_norm[i,2], np.multiply(X_norm[i,4], X_norm[i,0]))) + np.multiply(W49, np.multiply(X_norm[i,2], np.multiply(X_norm[i,4], X_norm[i,1])))  + b
        Y_pred = Y_pred_norm * Y_train_sigma + Y_train_mu
        #print("Y_pred is",Y_pred)
        #print("Y is",Y[i])
        difference = Y_pred - Y[i]
        deviation = np.divide(difference,Y[i])
        #deviation = (Y_pred_norm - Y[i])/Y[i]
        #print("difference is",difference)
        #print("deviation is",deviation)
        #if deviation < 0.1 and deviation > -0.1:
        #if deviation < 0.5 and deviation > -0.5:
        #if deviation < 0.1 and deviation > -0.1:
        if deviation < accuracy_tolerance and deviation > (-1 * accuracy_tolerance):
            num_accurate_datasets = num_accurate_datasets + 1
        else:
            mispredictedindices.append(i)
    accuracy = num_accurate_datasets/num_total_datasets


    return num_accurate_datasets,num_total_datasets, accuracy, mispredictedindices

# Normalize all of the features so that they're on the same numeric scale.
# Not doing this can lead to errors in the training process.
def normalize_features(dataset):
    n = dataset.shape[1]
    dataset_normalized = np.zeros((dataset.shape[0],dataset.shape[1]))
    mu = np.zeros((dataset.shape[1]))
    sigma = np.zeros((dataset.shape[1]))
    for i in range(n):
        mu[i] = np.mean(dataset[:,i],axis=0)
        sigma[i] = np.std(dataset[:,i],axis=0)
        dataset_normalized[:,i] = (dataset[:,i] - mu[i])/sigma[i]
    return dataset_normalized, mu, sigma

def process_input_data():

    wb = openpyxl.load_workbook("cpuint2000.xlsx")
    HPSheet = wb.get_sheet_by_name("Sheet3")

    baseline = []
    num_cores = []
    num_chips = []
    freq = []
    l1icache = []
    l1dcache = []
    l2cache = []
    l3 = []
    memory = []
    memory_freq= []

    baseline_train = []
    num_cores_train = []
    num_chips_train = []
    freq_train = []
    l1icache_train = []
    l1dcache_train = []
    l2cache_train = []
    l3_train = []
    memory_train = []
    memory_freq_train = []

    for col in HPSheet['B']:
        if col.value != "Baseline" :
            baseline.append(col.value)

    for col in HPSheet['C']:
        if col.value != "cores" :
            num_cores.append(col.value)

    for col in HPSheet['D']:
        if col.value != "chips" :
            num_chips.append(col.value)

    for col in HPSheet['F']:
        if col.value != "Processor MHz" :
            freq.append(col.value)

    for col in HPSheet['G']:
        if col.value != "L1 Icache" :
            l1icache.append(col.value)

    for col in HPSheet['H']:
        if col.value != "L1 Dcache" :
            l1dcache.append(col.value)

    for col in HPSheet['J']:
        if col.value != "L2" :
            l2cache.append(col.value)

    for col in HPSheet['L']:
        if col.value != "L3" :
            l3.append(col.value)

    for col in HPSheet['M']:
        if col.value != "Memory" :
            memory.append(col.value)

    for col in HPSheet['N']:
        if col.value != "DDRFrequency" :
            memory_freq.append(col.value)

    for i in range(len(baseline)):
        if baseline[i] != 0:
            baseline_train.append(baseline[i])
            num_cores_train.append(num_cores[i])
            num_chips_train.append(num_chips[i])
            freq_train.append(freq[i])
            l1icache_train.append(l1icache[i])
            l1dcache_train.append(l1dcache[i])
            l2cache_train.append(l2cache[i])
            l3_train.append(l3[i])
            memory_train.append(memory[i])
            memory_freq_train.append(memory_freq[i])

    X_data_temp = np.zeros((len(baseline),9))
    Y_data_temp = np.zeros((len(baseline),1))
    X_data_temp[:,0] = num_cores_train
    X_data_temp[:,1] = num_chips_train
    X_data_temp[:,2] = freq_train
    X_data_temp[:,3] = l1icache_train
    X_data_temp[:,4] = l1dcache_train
    X_data_temp[:,5] = l2cache_train
    X_data_temp[:,6] = l3_train
    X_data_temp[:,7] = memory_train
    X_data_temp[:,8] = memory_freq_train
    Y_data_temp[:,0] = baseline_train

    return X_data_temp,Y_data_temp

def create_train_cv_test_data(Xdata, Ydata):

    indicesList=[]
    for i in range(Xdata.shape[0]): 
        indicesList.append(i)
    
    np.random.seed(1)
    tf.set_random_seed(1)
    seed=1
    np.random.shuffle(indicesList)

    #print("indices is",indicesList)

    m = Xdata.shape[0]
    m_train = math.ceil(0.9 * m)
    m_cv = math.ceil(0.05 * m) #Floor so that m_test is never 0
    m_test = m - m_train - m_cv 
    print("m is",m)
    print("m_train is",m_train)
    print("m_cv is",m_cv)
    print("m_test is",m_test)
    X_train = np.zeros((m_train,9))
    X_cv = np.zeros((m_cv,9))
    X_test = np.zeros((m_test,9))
    Y_train = np.zeros((m_train,1))
    Y_cv = np.zeros((m_cv,1))
    Y_test = np.zeros((m_test,1))
    
    j = 0
    for i in range(m_train):
        index = indicesList[i]
        X_train[j,:] = Xdata[index,:]
        Y_train[j,:] = Ydata[index,:]
        j = j+1
    j = 0
    for i in range(m_cv):
        index = indicesList[m_train+i]
        X_cv[j,:] = Xdata[index,:]
        Y_cv[j,:] = Ydata[index,:]
        j = j+1
    j = 0
    for i in range(m_test):
        index = indicesList[m_train+m_cv+i]
        X_test[j,:] = Xdata[index,:]
        Y_test[j,:] = Ydata[index,:]
        j = j+1


    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test, indicesList

Xdata,Ydata = process_input_data()

print("Shape of Xdata is",Xdata.shape)
print("Shape of Ydata is",Ydata.shape)

shuffledindicesList = []
X_train_orig,Y_train_orig, X_cv_orig,Y_cv_orig, X_test_orig,Y_test_orig, shuffledindicesList = create_train_cv_test_data(Xdata,Ydata)
#X_train,Y_train, X_cv,Y_cv, X_test,Y_test, shuffledindicesList = create_train_cv_test_data(Xdata,Ydata)

m = X_train_orig.shape[0]

print("number of training examples are",m)
print("number of cv examples are",X_cv_orig.shape[0])
print("number of test examples are",X_test_orig.shape[0])

X_train, X_train_mu, X_train_sigma = normalize_features(X_train_orig)
Y_train, Y_train_mu, Y_train_sigma = normalize_features(Y_train_orig)

print("X_train_mu is",X_train_mu)
print("features normalized")


#Create placeholders
X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
X3 = tf.placeholder(tf.float32)
X4 = tf.placeholder(tf.float32)
X5 = tf.placeholder(tf.float32)
X6 = tf.placeholder(tf.float32)
X7 = tf.placeholder(tf.float32)
X8 = tf.placeholder(tf.float32)
X9 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

np.random.seed(1)
tf.set_random_seed(1)
seed=1
tf.set_random_seed(1)
'''
W1 = tf.get_variable("W1", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W2 = tf.get_variable("W2", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W3 = tf.get_variable("W3", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W4 = tf.get_variable("W4", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W5 = tf.get_variable("W5", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W6 = tf.get_variable("W6", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W7 = tf.get_variable("W7", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W8 = tf.get_variable("W8", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W9 = tf.get_variable("W9", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W10 = tf.get_variable("W10", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W11 = tf.get_variable("W11", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W12 = tf.get_variable("W12", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
W13 = tf.get_variable("W13", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b = tf.get_variable("b", [1], initializer=tf.contrib.layers.xavier_initializer(seed=1))


W1 = tf.Variable(tf.random_normal([1]))
W2 = tf.Variable(tf.random_normal([1]))
W3 = tf.Variable(tf.random_normal([1]))
W4 = tf.Variable(tf.random_normal([1]))
W5 = tf.Variable(tf.random_normal([1]))
W6 = tf.Variable(tf.random_normal([1]))
W7 = tf.Variable(tf.random_normal([1]))
W8 = tf.Variable(tf.random_normal([1]))
W9 = tf.Variable(tf.random_normal([1]))
W10 = tf.Variable(tf.random_normal([1]))
W11 = tf.Variable(tf.random_normal([1]))
W12 = tf.Variable(tf.random_normal([1]))
W13 = tf.Variable(tf.random_normal([1]))
#b = tf.Variable(tf.random_normal([1]))
b = tf.get_variable(name="b", shape=[1], initializer=tf.zeros_initializer()) 
'''

W1 = tf.Variable(tf.random_normal([1], stddev=0.05))
W2 = tf.Variable(tf.random_normal([1], stddev=0.05))
W3 = tf.Variable(tf.random_normal([1], stddev=0.05))
W4 = tf.Variable(tf.random_normal([1], stddev=0.05))
W5 = tf.Variable(tf.random_normal([1], stddev=0.05))
W6 = tf.Variable(tf.random_normal([1], stddev=0.05))
W7 = tf.Variable(tf.random_normal([1], stddev=0.05))
W8 = tf.Variable(tf.random_normal([1], stddev=0.05))
W9 = tf.Variable(tf.random_normal([1], stddev=0.05))
W10 = tf.Variable(tf.random_normal([1], stddev=0.05))
W11 = tf.Variable(tf.random_normal([1], stddev=0.05))
W12 = tf.Variable(tf.random_normal([1], stddev=0.05))
W13 = tf.Variable(tf.random_normal([1], stddev=0.05))
W14 = tf.Variable(tf.random_normal([1], stddev=0.05))
W15 = tf.Variable(tf.random_normal([1], stddev=0.05))
W16 = tf.Variable(tf.random_normal([1], stddev=0.05))
W17 = tf.Variable(tf.random_normal([1], stddev=0.05))
W18 = tf.Variable(tf.random_normal([1], stddev=0.05))
W19 = tf.Variable(tf.random_normal([1], stddev=0.05))
W20 = tf.Variable(tf.random_normal([1], stddev=0.05))
W21 = tf.Variable(tf.random_normal([1], stddev=0.05))
W22 = tf.Variable(tf.random_normal([1], stddev=0.05))
W23 = tf.Variable(tf.random_normal([1], stddev=0.05))
W24 = tf.Variable(tf.random_normal([1], stddev=0.05))
W25 = tf.Variable(tf.random_normal([1], stddev=0.05))
W26 = tf.Variable(tf.random_normal([1], stddev=0.05))
W27 = tf.Variable(tf.random_normal([1], stddev=0.05))
W28 = tf.Variable(tf.random_normal([1], stddev=0.05))
W29 = tf.Variable(tf.random_normal([1], stddev=0.05))
W30 = tf.Variable(tf.random_normal([1], stddev=0.05))
W31 = tf.Variable(tf.random_normal([1], stddev=0.05))
W32 = tf.Variable(tf.random_normal([1], stddev=0.05))
W33 = tf.Variable(tf.random_normal([1], stddev=0.05))
W34 = tf.Variable(tf.random_normal([1], stddev=0.05))
W35 = tf.Variable(tf.random_normal([1], stddev=0.05))
W36 = tf.Variable(tf.random_normal([1], stddev=0.05))
W37 = tf.Variable(tf.random_normal([1], stddev=0.05))
W38 = tf.Variable(tf.random_normal([1], stddev=0.05))
W39 = tf.Variable(tf.random_normal([1], stddev=0.05))
W40 = tf.Variable(tf.random_normal([1], stddev=0.05))
W41 = tf.Variable(tf.random_normal([1], stddev=0.05))
W42 = tf.Variable(tf.random_normal([1], stddev=0.05))
W43 = tf.Variable(tf.random_normal([1], stddev=0.05))
W44 = tf.Variable(tf.random_normal([1], stddev=0.05))
W45 = tf.Variable(tf.random_normal([1], stddev=0.05))
W46 = tf.Variable(tf.random_normal([1], stddev=0.05))
W47 = tf.Variable(tf.random_normal([1], stddev=0.05))
W48 = tf.Variable(tf.random_normal([1], stddev=0.05))
W49 = tf.Variable(tf.random_normal([1], stddev=0.05))
W50 = tf.Variable(tf.random_normal([1], stddev=0.05))
W51 = tf.Variable(tf.random_normal([1], stddev=0.05))
W52 = tf.Variable(tf.random_normal([1], stddev=0.05))
W53 = tf.Variable(tf.random_normal([1], stddev=0.05))
W54 = tf.Variable(tf.random_normal([1], stddev=0.05))
W55 = tf.Variable(tf.random_normal([1], stddev=0.05))
W56 = tf.Variable(tf.random_normal([1], stddev=0.05))
W57 = tf.Variable(tf.random_normal([1], stddev=0.05))
W58 = tf.Variable(tf.random_normal([1], stddev=0.05))
W59 = tf.Variable(tf.random_normal([1], stddev=0.05))
W60 = tf.Variable(tf.random_normal([1], stddev=0.05))
W61 = tf.Variable(tf.random_normal([1], stddev=0.05))
W61 = tf.Variable(tf.random_normal([1], stddev=0.05))
W62 = tf.Variable(tf.random_normal([1], stddev=0.05))
W63 = tf.Variable(tf.random_normal([1], stddev=0.05))
W64 = tf.Variable(tf.random_normal([1], stddev=0.05))
W65 = tf.Variable(tf.random_normal([1], stddev=0.05))
W66 = tf.Variable(tf.random_normal([1], stddev=0.05))
W67 = tf.Variable(tf.random_normal([1], stddev=0.05))
b = tf.get_variable(name="b", shape=[1], initializer=tf.zeros_initializer()) 
#b = tf.Variable(tf.random_normal([1], stddev=0.1))
#b = tf.Variable(tf.random_normal([1], stddev=0.1))

Y9 = W62*X3*X9*X1 + W63*X3*X9*X2 + W64*X3*X9*X4 + W65*X3*X9*X5 + W66*X3*X9*X6 + W67*X3*X9*X7

Y8 = W55*X8*X9*X3 + W56*X8*X9*X4 + W57*X8*X9*X5 + W58*X8*X9*X6 + W59*X8*X9*X7 + W60*X8*X9*X1 + W61*X8*X9*X2

Y7 = W52*X2*X4 + W53*X2*X9 + W54*X4*X9

Y6 = W49*X8*X2 + W50*X8*X4 + W51*X8*X9

Y5 = W45*X7*X2 + W46*X7*X4 + W47*X7*X8 + W48*X7*X9

Y4 = W40*X5*X2 + W41*X5*X4 + W42*X5*X7 + W43*X5*X8 + W44*X5*X9

Y3 = W34*X6*X2 + W35*X6*X4 + W36*X6*X5 + W37*X6*X7 + W38*X6*X8 + W39*X6*X9

Y2 = W31*X1*X1 + W32*X2*X2 + W33*X8*X9

Y1 = W1*X1 + W2*X2 + W3*X3 + W4*X4 + W5*X5 + W6*X6 + W7*X7 + W8*X8 + W9*X9 + W10*X3*X4 + W11*X3*X5 + W12*X3*X6 + W13*X3*X7 + W14*X3*X8 + W15*X3*X9 + W16*X3*X3 + W17*X4*X4 + W18*X5*X5 + W19*X6*X6 + W20*X7*X7 + W21*X8*X8 + W22*X9*X9 + W23*X1*X2 + W24*X1*X3 + W25*X1*X4 + W26*X1*X5 + W27*X1*X6 + W28*X1*X7 + W29*X1*X8 + W30*X1*X9 + b

Y_pred = Y1 + Y2 + Y3 + Y4 + Y5 + Y6 + Y7 + Y8 + Y9

#Y_pred = W1*X1 + W2*X2 + W3*X3 + W4*X4 + W5*X5 + W6*X6 + W7*X7 + W8*X8 + W9*X9 + b

#Y_pred = W1*X1 + W2*X2 + W3*X3 + W4*X4 + W5*X5 + W6*X6 + W7*X7 + W8*X8 + W9*X1*X1 + W10*X2*X2 + W11*X3*X3 + W12*X4*X4 + W13*X5*X5 + W14*X6*X6 + W15*X7*X7 + W16*X8*X8 + W17*X3*X4 + W18*X3*X5 + W19*X3*X6 + W20*X3*X7 + W21*X3*X8 + W22*X8*X4 + W23*X8*X5 + W24*X8*X6 + W25*X8*X7  + W26*X7*X6 + W27*X7*X5 + W28*X7*X4 + W29*X6*X4 + W30*X6*X5 + W31*X6*X7 + W32*X4*X5 + W33*X1*X3 + W34*X1*X4 + W35*X1*X5 + W36*X1*X6 + W37*X1*X7 + W38*X1*X8 + W39*X2*X1 + W40*X2*X4 + W41*X2*X5 + W42*X2*X6 + W43*X2*X7 + W44*X2*X8 + W45*X3*X5*X6 + W46*X3*X5*X7 + W47*X3*X5*X8 + W48*X3*X5*X1 + W49*X3*X5*X2 + b
#Y_pred = W1*X1 + W2*X2 + W3*X3 + W4*X4 + W5*X5 + W6*X6 + W7*X7 + W8*X8 + W9*X1*X1 + W10*X2*X2 + W11*X3*X3 + W12*X4*X4 + W13*X5*X5 + W14*X6*X6 + W15*X7*X7 + W16*X8*X8 + W17*X3*X4 + W18*X3*X5 + W19*X3*X6 + W20*X3*X7 + W21*X3*X8 + W22*X8*X4 + W23*X8*X5 + W24*X8*X6 + W25*X8*X7  + W26*X7*X6 + W27*X7*X5 + W28*X7*X4 + W29*X6*X4 + W30*X6*X5 + W31*X6*X7 + W32*X4*X5 + W33*X1*X3 + W34*X1*X4 + W35*X1*X5 + W36*X1*X6 + W37*X1*X7 + W38*X1*X8 + W39*X2*X1 + W40*X2*X4 + W41*X2*X5 + W42*X2*X6 + W43*X2*X7 + W44*X2*X8 + W45*X3*X5*X6 + W46*X3*X5*X7 + W47*X3*X5*X8 + W48*X3*X5*X1 + W49*X3*X5*X2 + W50*X9 + b
#Y_pred = W1*X1 + W2*X2 + W3*X3 + W4*X4 + W5*X5 + W6*X6 + W7*X7 + W8*X8 + W9*X1*X1 + W10*X2*X2 + W11*X3*X3 + W12*X4*X4 + W13*X5*X5 + W14*X6*X6 + W15*X7*X7 + W16*X8*X8 + W17*X3*X4 + W18*X3*X5 + W19*X3*X6 + W20*X3*X7 + W21*X3*X8 + b
#Y_pred = W1*X1 + W2*X2 + W3*X3 + W4*X4 + W5*X5 + W6*X6 + W7*X7 + W8*X8 + b

cost = tf.reduce_mean(tf.square(Y_pred - Y))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate, beta1=0.9,beta2=0.999,epsilon=adam_epsilon).minimize(cost)


#Train the Model

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
costs = []

min_cost = 0
min_cost_epoch = 0
#temp_cost = 0
epoch_cost = 0
for epoch in range(num_epochs):
    epoch_cost = 0
    temp_cost = 0
    for i in range(m):
        _,temp_cost = sess.run([optimizer,cost], feed_dict={X1:X_train[i,0], X2:X_train[i,1], X3:X_train[i,2],  X4:X_train[i,3], X5:X_train[i,4], X6:X_train[i,5], X7:X_train[i,6], X8:X_train[i,7], X9:X_train[i,8], Y:Y_train[i,0]})
        epoch_cost = epoch_cost + temp_cost/m
    costs.append(epoch_cost)
    print("Epoch %d Cost:%f"%(epoch,epoch_cost))
    if epoch%10==0:
        print("Epoch %d Cost:%f"%(epoch,epoch_cost))
    if epoch==1:
        min_cost = epoch_cost
        min_cost_epoch = epoch
    if epoch_cost<min_cost:
        min_cost = epoch_cost
        min_cost_epoch = epoch

plt.plot(costs)
plt.savefig('costs.png')
#print("W3 is %f",sess.run(W3))
#print("b3 is ",sess.run(b3))
print("min_cost is ",min_cost)
print("min_cost_epoch is ",min_cost_epoch)
print("final cost is ",epoch_cost)
#sess.close()


# Dump the parameters into params.json
params = {"W1": str(sess.run(W1)), "W2": str(sess.run(W2)), "W3": str(sess.run(W3)), "W4": str(sess.run(W4)), "W5": str(sess.run(W5)), "W6": str(sess.run(W6)), "W7": str(sess.run(W7)), "W8": str(sess.run(W8)), "W9":str(sess.run(W9)), "W10":str(sess.run(W10)), "W11":str(sess.run(W11)), "W12":str(sess.run(W12)), "W13":str(sess.run(W13)), "W14":str(sess.run(W14)), "W15":str(sess.run(W15)), "W16":str(sess.run(W16)), "W17":str(sess.run(W17)), "W18":str(sess.run(W18)), "W19":str(sess.run(W19)), "W20":str(sess.run(W20)), "W21":str(sess.run(W21)), "W22":str(sess.run(W22)), "W23":str(sess.run(W23)), "W24":str(sess.run(W24)), "W25":str(sess.run(W25)), "W26":str(sess.run(W26)), "W27":str(sess.run(W27)), "W28":str(sess.run(W28)), "W29":str(sess.run(W29)), "W30":str(sess.run(W30)), "W31":str(sess.run(W31)), "W32":str(sess.run(W32)) , "W33":str(sess.run(W33)) , "W34":str(sess.run(W34)) , "W35":str(sess.run(W35)) , "W36":str(sess.run(W36)) , "W37":str(sess.run(W37)) , "W38":str(sess.run(W38)) , "W39":str(sess.run(W39)) , "W40":str(sess.run(W40)) , "W41":str(sess.run(W41)) , "W42":str(sess.run(W42)) , "W43":str(sess.run(W43)) , "W44":str(sess.run(W44)) ,"W45":str(sess.run(W45)) , "W46":str(sess.run(W46)) , "W47":str(sess.run(W47)) , "W48":str(sess.run(W48)) , "W49":str(sess.run(W49)) , "W50":str(sess.run(W50)) , "W51":str(sess.run(W51)) , "W52":str(sess.run(W52)) , "W53":str(sess.run(W53)) , "W54":str(sess.run(W54)) , "W55":str(sess.run(W55)) , "W56":str(sess.run(W56)) , "W57":str(sess.run(W57)) , "W58":str(sess.run(W58)) , "W59":str(sess.run(W59)) , "W60":str(sess.run(W60)) , "W61":str(sess.run(W61)) , "W62":str(sess.run(W62)) , "W63":str(sess.run(W63)) , "W64":str(sess.run(W64)) , "W65":str(sess.run(W65)) , "W66":str(sess.run(W66)) , "W67":str(sess.run(W67)) , "b": str(sess.run(b)), "X_train_mu0": str(X_train_mu[0]), "X_train_mu1": str(X_train_mu[1]), "X_train_mu2": str(X_train_mu[2]), "X_train_mu3": str(X_train_mu[3]), "X_train_mu4": str(X_train_mu[4]), "X_train_mu5": str(X_train_mu[5]), "X_train_mu6": str(X_train_mu[6]), "X_train_mu7": str(X_train_mu[7]), "X_train_mu8": str(X_train_mu[8]), "X_train_sigma0": str(X_train_sigma[0]), "X_train_sigma1": str(X_train_sigma[1]), "X_train_sigma2": str(X_train_sigma[2]), "X_train_sigma3": str(X_train_sigma[3]), "X_train_sigma4": str(X_train_sigma[4]), "X_train_sigma5": str(X_train_sigma[5]), "X_train_sigma6": str(X_train_sigma[6]), "X_train_sigma7": str(X_train_sigma[7]), "X_train_sigma8": str(X_train_sigma[8]), "Y_train_mu": str(Y_train_mu), "Y_train_sigma": str(Y_train_sigma) }
with open('params.json', 'w') as outfile:
    json.dump(params, outfile)


mispredicted_train_indices = []

X_train_accurate_datasets, X_train_total_datasets, X_train_accuracy, mispredicted_train_indices = get_accuracy(X_train_orig, Y_train_orig)
print("X_train: ",X_train_accurate_datasets,"/",X_train_total_datasets,"accuracy:",X_train_accuracy)

mispredicted_train_indices_hash = {"mispredicted": str(mispredicted_train_indices)}
with open('mispredicted_indices.json', 'w') as outfile:
    json.dump(mispredicted_train_indices_hash, outfile)

#print("mispredicted indices are")
#for i in range(len(mispredicted_train_indices)):
#    tempi = mispredicted_train_indices[i]
#    print(" ",shuffledindicesList[tempi])


X_cv_accurate_datasets, X_cv_total_datasets, X_cv_accuracy, mispredicted_cv_indices = get_accuracy(X_cv_orig, Y_cv_orig)
print("X_cv: ",X_cv_accurate_datasets,"/",X_cv_total_datasets,"accuracy:",X_cv_accuracy)

X_test_accurate_datasets, X_test_total_datasets, X_test_accuracy, mispredicted_test_indices = get_accuracy(X_test_orig, Y_test_orig)
print("X_test: ",X_test_accurate_datasets,"/",X_test_total_datasets,"accuracy:",X_test_accuracy)



