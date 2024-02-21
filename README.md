# Activation-function_implementation

1. <h3>Theoretical Understanding</h3>
i) Explain the Activation Function, including its equation and graph.


The activation function in neural networks serves to introduce non-linearity into the output of a neuron, allowing the network to learn complex patterns and relationships in the data. It determines whether a neuron should be activated or not based on the weighted sum of its inputs plus a bias term.

Equation:

The output or activation (a) of a neuron is calculated using the following equation:

a = f(w1x1 + w2x2 + w3x3 + ... + wixi + b)

Where:

a =the activation of the neuron,

f=the activation function,

wi = the weights associated with the respective inputs xi,

xi = the input values to the neuron,

b= the bias term.

<h2>Activation Functions and their Derivatives</h2>

![306724352-0bcb4fd7-8294-4713-849f-1786fc657afe](https://github.com/monirakash/activation_function_implimentation/assets/82873473/d58a3b09-eaa4-4f4d-991f-c462c406899a)


<h2>Some commonly-used activation functions in neural networks are:</h2>

<h3>1.Sigmoid activation function</h3>

Features:

a.The sigmoid function has an s-shaped graph.

b.This is a non-linear function.

c.The sigmoid function converts its input into a probability value between 0 and 1.

d.It converts large negative values towards 0 and large positive values towards 1.

e.It returns 0.5 for the input 0. The value 0.5 is known as the threshold value which can decide that a given input belongs to what type of two classes.

code:
```

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Test
x_values = np.linspace(-10, 10, 100)  # create 100 points between -10 and 10
y_values = sigmoid(x_values)

# Plotting
import matplotlib.pyplot as plt

plt.plot(x_values, y_values)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

 ```
![306721925-701217bb-d877-43f4-ad8a-456daf251cc4](https://github.com/monirakash/activation_function_implimentation/assets/82873473/c9328684-e3fe-4354-8ab5-977201e7da56)


 <h2>2.Hyperbolic Tangent (tanh) Function</h2>
<p>a.It compresses the output between -1 and 1.</p>
<p>b.Also suffers from the vanishing gradient problem, but less so than the sigmoid.</p>

code:
```
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the tanh function
def tanh_function(x):
    return np.tanh(x)

# Generate an array of values from -10 to 10 to represent our x-axis
x = np.linspace(-10, 10, 400)

# Compute tanh values for each x
y = tanh_function(x)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='tanh(x)', color='blue')
plt.title('Hyperbolic Tangent Function (tanh)')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Setting the x and y axis limits
plt.axhline(y=0, color='black',linewidth=0.5)
plt.axvline(x=0, color='black',linewidth=0.5)
plt.legend()
plt.show()

```
![306722115-05ac666f-e20e-413b-89c8-ac609ec9a920](https://github.com/monirakash/activation_function_implimentation/assets/82873473/f4e66866-6030-4839-83dc-7878c5102aa0)

<h2>3.Rectified Linear Unit (ReLU)</h2>

<p>a.Formula: (f(x) = max(0, x))</p>
<p>b.Popular and widely used.</p>
<p>c.Introduces non-linearity.</p>

<b>Problems</b>: 
<p>Dying ReLU problem where neurons can sometimes get stuck during training and not activate at all.
</p>

Code:
```
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# Generate data
x = np.linspace(-10, 10, 400)
y = relu(x)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='ReLU Function', color='blue')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid()
plt.legend()
plt.show()
```
![306722323-90a20353-d8bd-4225-968b-386ff7baa402](https://github.com/monirakash/activation_function_implimentation/assets/82873473/21f4d0db-ba8c-4d2b-90e8-7f127878c138)

<h2>4.Leaky ReLU </h2>
<p>a.Variation of ReLU</p>.
<p>b.Tries to fix the Dying ReLU problem by allowing small negative values.</p>
Code:

```
import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Generate data
x = np.linspace(-10, 10, 400)
y = leaky_relu(x)

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Leaky ReLU', color='blue')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid()
plt.legend()
plt.show()

```
![306722504-436c3857-a365-4464-b93e-00371bfb930c](https://github.com/monirakash/activation_function_implimentation/assets/82873473/09de9cd5-de6f-4afe-8e07-dc82b234fa67)

<h2>5.Exponential Linear Unit (ELU)</h2>

<p>a.Tries to make the mean activations closer to zero which speeds up training.</p>
<p>b.Mitigates the dying ReLU problem.</p>

Code:

```
#Step 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
#Step 2: Define the ELU function
def elu(x, alpha=1.0):
    """ELU activation function."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

#Step 3: Visualize the ELU function
# Generate data
x = np.linspace(-10, 10, 400)
y = elu(x)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='ELU function', color='blue')
plt.title('Exponential Linear Unit (ELU) Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()
plt.show()

```

![306722689-98b539fa-bd52-4ba5-92bd-c53a8ae24f57](https://github.com/monirakash/activation_function_implimentation/assets/82873473/a97dc07c-298b-41ef-8fc1-ff14d7dc1421)


<h2>Softmax Activation Function</h2>
<p>Used in the output layer of the classifier where we are trying to attain the probabilities to define the class of each input.
</p>

Code:

```
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Example input
x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(outputs)

labels = ['x1', 'x2', 'x3']
plt.bar(labels, outputs)
plt.ylabel('Probability')
plt.title('Softmax Activation Output')
plt.show()

```

![304818036-13c90af9-1d58-475e-a1d7-9f45bc864d76](https://github.com/monirakash/activation_function_implimentation/assets/82873473/469d874d-0c6e-4168-a686-966652a3ded8)



3. How to Choose an Activation Function
<b>Understand the Purpose:</b>

<p>Classification: For binary classification, sigmoid is common in the output layer, while softmax is standard for multi-class tasks. For hidden layers, ReLU and its variants often work well.
</p>
<p>Regression: Linear or identity functions are typical for the output layer, with ReLU-based functions in hidden layers.</p>

<b>Avoid Saturating Activations for Deep Networks:</b>

Functions like the sigmoid and tanh can saturate (output values close to their min/max) causing vanishing gradient problems, especially in deep networks.
ReLU and its variants are often preferred due to their non-saturating nature.

<b>Address Dying Units with Modified ReLUs:</b>

Vanilla ReLU can cause dying units (neurons that stop learning) because they output zero for negative inputs.
Variants such as Leaky ReLU, Parametric ReLU, and Exponential Linear Unit (ELU) can alleviate this by allowing small negative outputs.

<b>Consider Computational Efficiency:</b>

The complexity of some functions (e.g., sigmoid or tanh) might not be suitable for real-time applications or when computational resources are limited.
ReLU and its variants offer computational advantages due to their simplicity.

<b>Factor in Initialization:</b>

Some activation functions, like Sigmoid or tanh, benefit from Xavier/Glorot initialization.
ReLU-based functions often work well with He initialization.

<b>Understand Task Specificity:</b>

Time-series: Activation functions that preserve certain properties of the input (like tanh, which preserves sign) might be beneficial.
Computer vision: ReLU and its variants have shown strong performance due to their capability to handle non-linearities.

<b>Consider Gradient Stability:</b>

It's vital to maintain a stable gradient flow, especially in deep networks.
Avoid functions that might result in exploding or vanishing gradients. For example, if using tanh or sigmoid, be wary of the network's depth and other hyperparameters.

<b>Custom or Novel Activations:</b>

Sometimes, a problem may require a custom activation function. Experimentation can help uncover if a tailored function provides benefits.
Stay updated with recent literature, as newer activations like Swish and Mish have emerged and shown promise.

<b>Safety with Sparsity:</b>

If sparsity (having more zeros in the output) is desired, functions like ReLU naturally induce it.
However, too much sparsity might not be always beneficial. Balance is key.

<b>Regularization Interplay:</b>

Consider how the activation function interacts with regularization techniques like dropout.
Some combinations might work synergistically, while others could hinder learning.

<b>Batch Normalization Impact:</b>

Using Batch Normalization can alleviate some issues tied to activation functions by normalizing layer outputs.
This can make the choice of activation function less critical, but it's still an essential consideration.

<b>Empirical Testing:</b>

Theoretical insights are helpful, but empirical testing on validation data is crucial.
Always benchmark different activation functions under the same conditions to determine the best fit.

<b>Understand the Output Range:</b>

Recognize the range of your desired output. For instance, using a ReLU in the output layer of a regression task might not be ideal if negative outputs are possible.

<b>Problem Constraints:</b>
In certain applications, the interpretability of a model might be vital. Some activation functions can make models more interpretable than others.

<b>Research and Community Consensus:</b>
Often, the broader machine learning community converges on certain best practices for specific tasks. Stay updated with the latest research.
