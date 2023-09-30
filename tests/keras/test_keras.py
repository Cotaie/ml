import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


model = Sequential(tf.keras.Input(shape=(2,)), [Dense(3,activation='sigmoid', name='layer1'), Dense(1, activation='sigmoid', name='layer2')])

model.summary()