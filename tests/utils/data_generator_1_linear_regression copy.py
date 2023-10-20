import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define parameters for the quadratic function
#a, b, c = 0, 3, 2

# Generate data
np.random.seed(0)  # For reproducibility
x1 = np.linspace(-50,0,500)
y1 = (-3)*x1 + 0 + np.random.normal(0, 2, x1.shape)
x2 = np.linspace(0, 50, 500)
y2 = 3 * x2 + 0 + np.random.normal(0, 2, x2.shape)
# x3 = np.linspace(50,100,500)
# y3 = 0*x1 + 200 + np.random.normal(0, 2, x3.shape)
# x4 = np.linspace(100,150,500)
# y4 = -b * x4 + 450 + np.random.normal(0, 2, x4.shape)
# x5 = np.linspace(150,200,500)
# y5 = 0*x1 + np.random.normal(0, 2, x5.shape)

# # Add noise to the data
# y_noisy = y + np.random.normal(0, 2, y.shape)
# y1_noisy = y1 + np.random.normal(0, 2, y1.shape)

df = pd.DataFrame({
    'x': x1,
    'y': y1,
})
df.to_csv('data_1_0db_1.csv', index=False)

df = pd.DataFrame({
    'x': x2,
    'y': y2,
})
df.to_csv('data_1_0db_1.csv', mode='a', header=False, index=False)

# df = pd.DataFrame({
#     'x': x3,
#     'y': y3,
# })
# df.to_csv('data_1_0db_1.csv', mode='a', header=False, index=False)

# df = pd.DataFrame({
#     'x': x4,
#     'y': y4,
# })
# df.to_csv('data_1_0db_1.csv', mode='a', header=False, index=False)

# df = pd.DataFrame({
#     'x': x5,
#     'y': y5,
# })
# df.to_csv('data_1_0db_1.csv', mode='a', header=False, index=False)

# Plot the data
plt.scatter(x1, y1, color='blue', s=8, label='Noisy Data')
plt.scatter(x2, y2, color='blue', s=8, label='Noisy Data')
#plt.scatter(x3, y3, color='blue', s=8, label='Noisy Data')
# plt.scatter(x4, y4, color='blue', s=8, label='Noisy Data')
# plt.scatter(x5, y5, color='blue', s=8, label='Noisy Data')
plt.legend()
plt.show()

# You can split this data into training and testing subsets to test your MLP.
