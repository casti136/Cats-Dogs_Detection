import tensorflow as tf
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D,Activation

X = []
Y = []

# Retrieve X Values
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("Y.pickle","rb")
Y = pickle.load(pickle_in)


#normalize X
Y = np.array(Y)
X = tf.keras.utils.normalize(X, axis=1)
#print(X[1])
#1
model = tf.keras.models.Sequential()
model.add(Conv2D(64,(3,3),input_shape = X.shape[1:])) #batchsize, 50 x 50 , 1 is rgb
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size =(2,2)))

#2
model.add(Conv2D(64,(3,3))) #batchsize, 50 x 50 , 1 is rgb
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size =(2,2)))
model.add(tf.keras.layers.Flatten())

#3
model.add(tf.keras.layers.Dense(64))

#Output layer
model.add(tf.keras.layers.Dense(1, activation =tf.nn.sigmoid))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X, Y, batch_size=32, epochs =5, validation_split=0.1)




