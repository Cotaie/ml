import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input,Dense
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam


data = pd.read_csv('data_fin_3.csv')

x1_values = data['X'].values
x2_values = data['Y'].values
label = data['Label'].values
X = [list(item) for item in zip(x1_values, x2_values)]
Y = [[item] for item in label]

X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)

model = Sequential(
    [Input(shape=(2,)),
     Dense(3, activation='sigmoid', name = 'layer1'),
     Dense(1, activation='sigmoid', name = 'layer2')]
)

model.summary()

model.compile(loss=BinaryCrossentropy(),
              optimizer=Adam(learning_rate=0.01))

model.fit(X,Y,epochs=1)

print(model.predict(np.array([[2,9]])))

plt.scatter(x1_values[label == 0], x2_values[label == 0], label='Class 0', alpha=0.5)
plt.scatter(x1_values[label == 1], x2_values[label == 1], label='Class 1', alpha=0.5)
plt.show()
