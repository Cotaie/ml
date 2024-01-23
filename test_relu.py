# import numpy as np
# import matplotlib.pyplot as plt

# # Define the individual ReLU functions
# def ReLU_1(x):
#     return np.maximum(0, x)

# def ReLU_2(x):
#     return np.maximum(0, -x)

# def ReLU_3(x):
#     return np.maximum(0, 2*x)

# # Define the combined function
# def Combined_Output(x):
#     return ReLU_1(x) + ReLU_2(x) + ReLU_3(x)

# # Create an array of x values from -2 to 2 with a step of 0.01
# x_values = np.arange(-2, 2, 0.01)

# # Calculate the output of each individual ReLU function and the combined function
# output_ReLU_1 = ReLU_1(x_values)
# output_ReLU_2 = ReLU_2(x_values)
# output_ReLU_3 = ReLU_3(x_values)
# output_Combined = Combined_Output(x_values)

# # Plotting individual ReLU functions
# plt.figure(figsize=(8, 6))
# plt.plot(x_values, output_ReLU_1, label='ReLU_1(x) = max(0, x)')
# plt.plot(x_values, output_ReLU_2, label='ReLU_2(x) = max(0, -x)')
# plt.plot(x_values, output_ReLU_3, label='ReLU_3(x) = max(0, 2x)')
# plt.xlabel('x')
# plt.ylabel('Output')
# plt.title('Individual ReLU Functions')
# plt.legend()
# plt.grid(True)
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.show()

# # Plotting the combined function
# plt.figure(figsize=(8, 6))
# plt.plot(x_values, output_Combined, label='Combined_Output(x)')
# plt.xlabel('x')
# plt.ylabel('Output')
# plt.title('Combined Function')
# plt.legend()
# plt.grid(True)
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the individual ReLU functions
def ReLU_1(x):
    return np.maximum(0, x)

def ReLU_2(x):
    return np.maximum(0, -x)

def ReLU_3(x):
    return np.maximum(0, x-1)

def ReLU_4(x):
    return np.maximum(0, 1-x)

# Define the combined function to approximate the parabola y = x²
def Parabola_Approximation(x):
    return ReLU_1(x) + ReLU_2(x) + ReLU_3(x) + ReLU_4(x)

# Create an array of x values from -2 to 2 with a step of 0.01
x_values = np.arange(-2, 2, 0.01)

# Calculate the output of the individual ReLU functions
output_ReLU_1 = ReLU_1(x_values)
output_ReLU_2 = ReLU_2(x_values)
output_ReLU_3 = ReLU_3(x_values)
output_ReLU_4 = ReLU_4(x_values)

# Calculate the output of the combined function
output_combined = Parabola_Approximation(x_values)

# Plotting the individual ReLU functions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_values, output_ReLU_1, label='ReLU_1(x) = max(0, x)')
plt.plot(x_values, output_ReLU_2, label='ReLU_2(x) = max(0, -x)')
plt.plot(x_values, output_ReLU_3, label='ReLU_3(x) = max(0, x — 1)')
plt.plot(x_values, output_ReLU_4, label='ReLU_4(x) = max(0, 1 — x)')
plt.xlabel('x')
plt.ylabel('Output')
plt.title('Individual ReLU Functions')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Plotting the combined function
plt.subplot(1, 2, 2)
plt.plot(x_values, output_combined, label='Parabola Approximation')
plt.xlabel('x')
plt.ylabel('Output')
plt.title('Approximating a Parabola using ReLU functions')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()