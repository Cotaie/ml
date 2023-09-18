import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the quadratic function
a, b, c = 1, -3, 2

# Generate data
np.random.seed(0)  # For reproducibility
x = np.linspace(-10, 10, 400)
y = a * x**2 + b * x + c

# Add noise to the data
noise = np.random.normal(0, 10, y.shape)
y_noisy = y + noise

# Plot the data
plt.scatter(x, y_noisy, color='blue', s=8, label='Noisy Data')
plt.plot(x, y, color='red', label='True Function')
plt.legend()
plt.show()

# You can split this data into training and testing subsets to test your MLP.
