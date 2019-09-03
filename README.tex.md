# Backpropagation continued

THIS PART OF THE SERIES IS STILL 🚧UNDER CONSTRUCTION🚧

This section explores a few of the problems (and solutions) to neural networks and backpropagation. 

We will polish our previous network model by applying a few new techniques and optimizations to speed up and stabilize the learning process. 

## This is part 3 of a series of github repos on neural networks

- [part 1 - simplest network](https://github.com/gokadin/ai-simplest-network)
- [part 2 - backpropagation](https://github.com/gokadin/ai-backpropagation)
- part 3 - backpropagation-continued (**you are here**)
- [part 4 - hopfield networks](https://github.com/gokadin/ai-hopfield-networks)

## Table of Contents

- [Theory](#theory)
  - [Local and global minima](#local-and-global-minima)
  - [Over fitting and generalization](#over-fitting-and-generalization)
  - [The vanishing gradient problem](#the-vanishing-gradient-problem)
    - [Batch normalization](#batch-normalization)
  - [Different types of gradient descent](#different-types-of-gradient-descent)
    - [Batch gradient descent](#batch-gradient-descent)
    - [Stochastic gradient descent](#stochastic-gradient-descent)
    - [Mini-batch gradient descent](#mini-batch-gradient-descent)
  - [Optimization functions](#optimization-functions)
    - [Momentum](#momentum)
    - [ADAM](#adam)
  - [Better weight initialization](#better-weight-initialization)
- [Code example](#code-example)
- [References](#references)

## Theory

### Local and global minima

The error function will usually have many local minima (green dots) in addition to its global minimum (red dot). The more shallow a local minimum is, the least optimized is the solution. 

![minima](readme-images/minima.jpg)

Gradient descent will always go towards the nearest minimum which will almost always be a local one. However, since error functions have high dimentionality (proportional to the number of weights), its local minima are very deep, if not almost as deep as the global one. 

### Over fitting and generalization

🚧UNDER CONSTRUCTION🚧

### The vanishing gradient problem

In a network with many hidden layers, the vanishing gradient problem is when the the gradient becomes extremely small the further back it goes towards the input layer. 

Consider the derivative of a non-linear activation function:

$$ f\prime(x) = f(x)(1 - f(x)) $$

$f\prime(x)$ approaches zero when $f(x)$ is close to either zero or one. This is then multipled within the chain rule of the gradient calculation and the more layers with non-linear activation functions it passes through, the more the gradient becomes closer to zero. 

This effectively means that layers closer to the output layer do most of the learning. 

The other extreme of this problem is the *exploding* gradient. This is when the gradient becomes extremely large, usually occuring in temporal backpropagation networks. 

#### Batch normalization

One technique used to reduce the vanishing gradient problem is to normalize the hidden layer outputs just as we normalize inputs before feeding them to the network. Example, if the input values are from 0 to 255, we can normalize them to be between 0 and 1. If we don't normalize the inputs, high values can destabilize the network and lead to the exploding gradient problem as well as slow the training speed substantially. 

This technique makes sure that the hidden layer activations don't go to the extremes (say 0 and 1), so that they don't diminish the gradient too much for the following layers. 

A normalized value $z$ is achieved by subtracting the mean $m$ and dividing by the standard deviation $s$: $z = \frac{y - m}{s} $

We then transform $z$ with two new trainable parameters $g$ and $b$: $(z * g) + b$

### Different types of gradient descent

#### Batch gradient descent

This is essentially what we've been doing in *part 1* and *part 2* of this series. We only update the network weights once all of the training associations have gone trough. 

One cycle through all of the associations is called an **epoch** and batch gradient descent updates the weights at the end of each epoch. 

![batch](readme-images/batch.jpg)

This has the advantage of updating the weights less frequently, making it more efficient, but it risks converging too soon to a less optimal function minimum. 

#### Stochastic gradient descent

Stochastic gradient descent is the opposite. Instead of updating the weights once at the end of each epoch, it updates them for each association. 

![stochastic](readme-images/stochastic.jpg)

This obviously makes it slower to run since we have more operations per training association. It can avoid the premature convergence problem of batch gradient descent since the variance over training epochs is higher, however this can equally make it harder for the network to converge towards a minimum. 

The *stochastic* part means that it shuffles the dataset at the beginning of each epoch. 

#### Mini-batch gradient descent

Mini-batch takes advantage of the strenghts of both methods by splitting the dataset into small batches and updating the weights after each batch is processed. 

Note that stochastic gradient descent is often used to mean mini-batch. In fact, the only difference between the two is the number of associations between weight updates. Mini-batch equally shuffles the dataset at the beginning of each epoch and then partitions the data into batches. 

![mini-batch](readme-images/mini-batch.jpg)

In the image above the dataset was split into mini batches of 2 associations each. A good batch size can be anything and is one of the parameters that needs to be frequentely tuned for optimal results. Keep in mind that larger batch sizes should compute faster. 

### Optimization functions

🚧UNDER CONSTRUCTION🚧

#### Momentum

🚧UNDER CONSTRUCTION🚧

#### ADAM

🚧UNDER CONSTRUCTION🚧

### Better weight initialization

🚧UNDER CONSTRUCTION🚧

## Code example

🚧UNDER CONSTRUCTION🚧

## References

- Artificial intelligence engines by James V Stone (2019)
- Complete guide on deep learning: http://neuralnetworksanddeeplearning.com/chap2.html
- Mini-batch gradient descent: https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
- Batch normalization: https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c