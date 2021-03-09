import matplotlib
import matplotlib.pyplot as plt
import keras as k
import numpy as np
import time,os
import tensorflow
######################################
(xtrain,ytrain),(xtest,ytest) = k.datasets.mnist.load_data()
######################################
xtrain[0].shape
######################################
xtrain = xtrain/256
ytrain = ytrain/256
######################################
xtrainflatten=xtrain.reshape(len(xtrain),28*28)
xtestflatten =xtest.reshape(len(xtest),28*28)
##########################################
xtrainflatten[0]
##########################################
model = k.Sequential(

    [k.layers.Dense(10,input_shape=(784,),activation='sigmoid')]

)

model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']

)

model.fit(xtrainflatten,ytrain,epochs=5)
##########################################
model.evaluate(xtestflatten,ytest)
###########################################
firstpred = model.predict(xtestflatten)
print(np.argmax(firstpred[9]))
y = np.argmax(firstpred[9])
print(y)
#############################################
