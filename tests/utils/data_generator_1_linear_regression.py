import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define parameters for the quadratic function
a, b, c = 0, 3, 2

# Generate data
np.random.seed(0)  # For reproducibility
x = np.linspace(-50, 50, 1000)
x1 = np.linspace(50,100,500)
y = a * x**2 + b * abs(x) + c
y1 = (-3) * x1 + 300

# Add noise to the data
y_noisy = y + np.random.normal(0, 2, y.shape)
y1_noisy = y1 + np.random.normal(0, 2, y1.shape)

df = pd.DataFrame({
    'x': x,
    'y': y_noisy,
})

# Save to CSV
df.to_csv('data_1_0db_1.csv', index=False)

df = pd.DataFrame({
    'x': x1,
    'y': y1_noisy,
})

df.to_csv('data_1_0db_1.csv', mode='a', header=False, index=False)

# Plot the data
plt.scatter(x, y_noisy, color='blue', s=8, label='Noisy Data')
plt.scatter(x1, y1_noisy, color='yellow', s=8, label='Noisy Data')
plt.plot(x, y, color='red', label='True Function')
plt.plot(x1, y1, color='red', label='True Function')
plt.legend()
plt.show()

# You can split this data into training and testing subsets to test your MLP.
