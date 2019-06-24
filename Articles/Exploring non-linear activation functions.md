# Exploring non-linear activation functions

An activation function is an important transformer in a neural network which enables a model to learn non-linearity. Given an input or set of inputs, the output of each node in a neural network is defined by weights and an activation function. Non-linear activation functions are important in neural network hidden units; without them, the output of a neural network is simply a linear function of the input. Except for some special cases (i.e. regression), non-linear activation functions are highly desirable. 

Similar to how the human brain works, an activation function determines whether or not a neuron (each node in a neural network) should fire a signal to a next neuron. Depending on the type of an activation function, the corresponding threshold and the boundary of the activation output differ greatly.

When ***z*** is the input to the activation function, the output can be written as ***g(z)***.

Typical examples of activation functions widely used today include:

### Sigmoid (logistic) function:

<p align="center"><img src="../images/sigmoid.png" width="350"></p>

- Non-linear function that is widely used in classification problems
- Good for output layer in binary classification
- Output value range in 0 and 1
- When ***z*** is very large or small, the slope becomes very small —> slow down the learning process ("gradient vanishing")

### Hyperbolic Tangent (Tanh) function:

<p align="center"><img src="../images/tanh.png" width="350"></p>

* Non-linear function that is similar to a sigmoid function
* Output value range in -1 and 1
* Gradient is stronger than a sigmoid function
* Almost always work better than sigmoid as it centers the data around 0 mean
* When ***z*** is very large or small, the slope becomes very small —> slow down the learning process ("gradient vanishing")

### Rectified Linear Unit (ReLU) function:

<p align="center"><img src="../images/relu.png" width="350"></p>

- Non-linear function that helps in the sparsity of the activation (computationally efficient compared to sigmoid and tanh)
- One of the most popular choice in today's deep learning models as it can avoid a vanishing gradient problem
- Dying ReLU problem: gradient can go towards 0, and if it does, no more learning takes place in several neurons
- Since the output is 0 for all negative input, their mean activation is greater zero —> causes bias shift for a unit in next layer and thus slow down the learning process ([ref](https://arxiv.org/pdf/1511.07289.pdf))

### Exponential Linear Unit (ELU) function:

<p align="center"><img src="../images/elu.png" width="350"></p>

* Variation of ReLU that resolves the dying ReLU problem
* With its negative activations, their mean activation is close to zero and thus speeds up the learning process and convergence. 

## Simple experiment on MNIST dataset

to be added..