# Backpropagation continued

This section explores a few of the problems (and solutions) to neural networks and backpropagation. 

## This is part 3 of a series of github repos on neural networks

- [part 1 - simplest network](https://github.com/gokadin/ai-simplest-network)
- [part 2 - backpropagation](https://github.com/gokadin/ai-backpropagation)
- part 3 - backpropagation-continued (**you are here**)
- [part 4 - hopfield networks](https://github.com/gokadin/ai-hopfield-networks)

## Table of Contents

- [Theory](#theory)
  - [Batch gradient descent](#batch-gradient-descent)
  - [Stochastic gradient descent](#stochastic-gradient-descent)
  - [Mini-batch gradient descent](#mini-batch-gradient-descent)
- [Code example](#code-example)
- [References](#references)

## Theory

🚧UNDER CONSTRUCTION🚧

Remaining subjects to be covered:

- momentum
- over fitting/generalization/early stopping point
- vanishing/exploding gradients
- local and global minima

### Batch gradient descent

This is essentially what we've been doing in *part 1* and *part 2* of this series. We only update the network weights once all of the training associations have gone trough. 

One cycle through all of the associations is called an **epoch** and batch gradient descent updates the weights at the end of each epoch. 

![batch](readme-images/batch.jpg)

This has the advantage of updating the weights less frequently, making it more efficient, but it risks converging too soon to a less optimal function minimum. 

### Stochastic gradient descent

Stochastic gradient descent is the opposite. Instead of updating the weights once at the end of each epoch, it updates them for each association. 

![stochastic](readme-images/stochastic.jpg)

This obviously makes it slower to run since we have more operations per training association. It can avoid the premature convergence problem of batch gradient descent since the variance over training epochs is higher, however this can equally make it harder for the network to converge on a minimum. 

The *stochastic* part means that it chooses a random association each time. Shuffling the order of associations in the dataset would also produce the same effect. 

### Mini-batch gradient descent

Mini-batch is a hybrid of batch and stochastic gradient descent. It takes advantage of the strenghts of both methods by splitting the dataset into small batches and updating the weights after each batch is processed. 

![mini-batch](readme-images/mini-batch.jpg)

In the image above the dataset was split into mini batches of 2 associations each. A good batch size can be anything and is one of the parameters that needs to be frequentely tuned for optimal results. Keep in mind that larger batch sizes should compute faster. 

## Code example

🚧UNDER CONSTRUCTION🚧

## References

- Artificial intelligence engines by James V Stone (2019)
- Complete guide on deep learning: http://neuralnetworksanddeeplearning.com/chap2.html
- https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/