import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural import Layer, Model

data = pd.read_csv('data_bin.csv')

x1_values = data['X'].values
x2_values = data['Y'].values
label = data['Label'].values
X = [list(item) for item in zip(x1_values, x2_values)]
Y = [[item] for item in label]

# Define the lines: y = mx + b
m1, b1 = -1, 10   # First line
# m2, b2 = 0, 2   # Second line
# m3, b3 = 1, 0   # Third line

#mod_arch = [2, Layer(3, activation="sigmoid"), Layer(1, activation="sigmoid")]
mod_arch = [2, Layer(1, activation="sigmoid")]
mod = Model(mod_arch)
mod.compile(loss='binary_crossentropy')
#mod.compile(loss='mse')
#mod._set_W_1()
#print(mod.predict(np.array([2, 10])))

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
normalized_X = (X - mean) / std

# print("mean: ", mean)
# print("std: ", std)

mod.fit(normalized_X, Y, epochs=10)
# for x, y in zip(X,Y):
#     print("predict: ", mod.predict(x), "should be: ", y)
norm_input = (np.array([6.050493541021657,0.9663189290801799]) - mean) / std
print(mod.predict(norm_input))



plt.scatter(x1_values[label == 0], x2_values[label == 0], label='Class 0', alpha=0.5)
plt.scatter(x1_values[label == 1], x2_values[label == 1], label='Class 1', alpha=0.5)
plt.plot([0, 10], [b1, 10 * m1 + b1], '-r')
# plt.plot([0, 10], [b2, 10 * m2 + b2], '-g')
# plt.plot([0, 10], [b3, 10 * m3 + b3], '-b')
plt.legend()
plt.show()
