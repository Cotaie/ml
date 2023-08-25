import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv('data_fin.csv')

x1_values = data['X'].values
x2_values = data['Y'].values
label = data['Label'].values
X = [list(item) for item in zip(x1_values, x2_values)]
Y = [[item] for item in label]
