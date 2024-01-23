import numpy as np
from neural.normalizations import Normalization
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import minimize

data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data_ramp.csv')
X = [item for item in data['x'].values]
Y = [item for item in data['y'].values]
plt.scatter(X, Y, label='input', color='blue')

def relu(z):
    return np.maximum(0, z)

X = x1 = np.linspace(-50,100,1500)
w0 = -1.5
b0 = 1
w1 = -2
b1 = 1

#plt.plot(X, [relu(w1*relu(w0*x + b0) + b1) for x in X], color='green')
#plt.plot(X, [relu(w0*x + b0) for x in X])
#plt.plot(X, [relu(w0*x + b0) + relu(w1*x + b1)for x in X], color='green')

def fct(X,Y):
    l = len(X)
    def fct_(x0):
        mse = 0
        b0 = x0[0]
        b1 = x0[1]
        w0 = x0[2]
        w1 = x0[3]
        for x,y in zip(X,Y):
            mse += np.square(relu(w1*x + b1) + relu(w0*x + b0) - y) / l
        return mse
    return fct_


f = fct(X,Y)

min_val = minimize(f,x0=(75, 150, 1, 1)).x

print("min val:", min_val)

#plt.plot(X, [relu(min_val[3]*relu(min_val[2]*x + min_val[0]) + min_val[1]) for x in X], color='red')
plt.plot(X, [relu(min_val[2]*x + min_val[0]) + relu(min_val[3]*x + min_val[1])for x in X], color='red')
plt.show()