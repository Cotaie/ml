import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Number of data points
N = 10000

# Generate random data points
x = np.random.rand(N) * 10  # Random x values between 0 and 10
y = np.random.rand(N) * 10  # Random y values between 0 and 10

# Define the lines: y = mx + b
m1, b1 = -1, 10   # First line
m2, b2 = 0, 2   # Second line
#m3, b3 = 1, 0   # Third line

# Classify data points
labels = np.where(
    #((y < m1 * x + b1) & (y > m2 * x + b2) & (y < m3 * x + b3)), 0, 1
    ((y < m1 * x + b1) & (y > m2 * x + b2)), 0, 1
    )

# Convert data to a pandas DataFrame for saving
df = pd.DataFrame({
    'x1': x,
    'x2': y,
    'y': labels
})

# Save to CSV
df.to_csv('data_2_db_1.csv', index=False)

# Plotting
plt.scatter(x[labels == 0], y[labels == 0], label='Class 0', alpha=0.5)
plt.scatter(x[labels == 1], y[labels == 1], label='Class 1', alpha=0.5)
plt.plot([0, 10], [b1, 10 * m1 + b1], '-r')
plt.plot([0, 10], [b2, 10 * m2 + b2], '-g')
#plt.plot([0, 10], [b3, 10 * m3 + b3], '-b')
plt.legend()
plt.show()
