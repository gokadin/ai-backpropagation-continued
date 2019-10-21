# Backpropagation continued

THIS PART OF THE SERIES IS STILL 🚧UNDER CONSTRUCTION🚧

This section explores a few of the problems (and solutions) to neural networks and backpropagation. 

We will polish our previous network model by applying a few new techniques and optimizations to speed up and stabilize the learning process. 

## This is part 3 of a series of github repos on neural networks

- [part 1 - simplest network](https://github.com/gokadin/ai-simplest-network)
- [part 2 - backpropagation](https://github.com/gokadin/ai-backpropagation)
- part 3 - backpropagation-continued (**you are here**)

## Table of Contents

- [Theory](#theory)

  - [Local and global minima](#local-and-global-minima)

  - [Over fitting and generalization](#over-fitting-and-generalization)

  - [The vanishing gradient problem](#the-vanishing-gradient-problem)

  - [More activation functions](#more-activation-functions)
  - [Sigmoid](#sigmoid)
    - [Tanh](#tanh)
    - [ReLU](#relu)
    - [Leaky ReLU](#leaky-relu)
    - [Softmax](#softmax)

  - [More error functions](#more-error-functions)

    - [Cross entropy](#cross-entropy)

  - [Different types of gradient descent](#different-types-of-gradient-descent)

    - [Batch gradient descent](#batch-gradient-descent)
    - [Stochastic gradient descent](#stochastic-gradient-descent)
    - [Mini-batch gradient descent](#mini-batch-gradient-descent)

  - [Optimization techniques](#optimization-techniques)
  - [Momentum](#momentum)
    - [Adam](#adam)

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

<p align="center"><img src="/tex/8269ff1c3dd8ffddf8ec3c46ac1de9e9.svg?invert_in_darkmode&sanitize=true" align=middle width=163.52750865pt height=16.438356pt/></p>

<img src="/tex/10524ebc8ff76449e78afaa538e66c95.svg?invert_in_darkmode&sanitize=true" align=middle width=36.51837584999999pt height=24.65753399999998pt/> approaches zero when <img src="/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/> is close to either zero or one. This is then multipled within the chain rule of the gradient calculation and the more layers with non-linear activation functions it passes through, the more the gradient becomes closer to zero. 

This effectively means that layers closer to the output layer do most of the learning. 

The other extreme of this problem is the *exploding* gradient. This is when the gradient becomes extremely large, usually occuring in temporal backpropagation networks. 

### More activation functions

#### Sigmoid

![sigmoid](readme-images/sigmoid.jpg)

<p align="center"><img src="/tex/3816927dd5a0c9923b367f125f3141a2.svg?invert_in_darkmode&sanitize=true" align=middle width=308.7797196pt height=34.3600389pt/></p>

As seen in the vanishing gradient section, the sigmoid function we've been using so far tends to kill the gradient when its output is close to zero or one. 

Another drawback of the sigmoid is that its not zero-centered, meaning its output is always positive. This makes the gradient either all positive or all negative, introducing zig-zagging dynamic in weight updates. 

Because of these issues, sigmoid is not recommended to be used anymore. Let's explore some of the other functions instead. 

#### Tanh

<p align="center"><img src="/tex/c42b7255a4dd125d8c28cbbeada65a55.svg?invert_in_darkmode&sanitize=true" align=middle width=305.4119409pt height=34.3600389pt/></p>

![tanh](readme-images/tanh.jpg)

Tanh is similar to the sigmoid function, except that it outpus in a range of <img src="/tex/43ca5ad9e1f094a31392f860ef481e5c.svg?invert_in_darkmode&sanitize=true" align=middle width=45.66218414999998pt height=24.65753399999998pt/>, making it zero-centered. However just like the sigmoid it also causes the vanishing gradient problem. 

#### ReLU

<p align="center"><img src="/tex/71bc6be083fe3373e011325434905609.svg?invert_in_darkmode&sanitize=true" align=middle width=356.29890779999994pt height=49.315569599999996pt/></p>

![relu](readme-images/relu.jpg)

ReLU or rectified linear unit is the most commonly used function in today's neural networks for hidden layers and was found to greatly accelerate convergence compare to sigmoid and tanh. 

It doesn't diminish the gradient down to almost zero at both extremes like the sigmoid and tanh functions and it's faster to compute since it's just taking the maximum value of zero and <img src="/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>. 

However it does have its issues: it's also not zero-centered and the gradients for negative inputs are zero. This can kill a node, since weights are not updated, sometimes causing a node to never fire again. 

#### Leaky ReLU

<p align="center"><img src="/tex/1faa102cbff1eda2ffab5fd7170486a2.svg?invert_in_darkmode&sanitize=true" align=middle width=406.5273729pt height=49.315569599999996pt/></p>

![leaky-relu](readme-images/lrelu.jpg)

Leaky ReLU tries to fix the killing tendencies of ReLU by introducing a small negative slope when <img src="/tex/949d3fe7fc31d082be4b1fbe3eb4ac89.svg?invert_in_darkmode&sanitize=true" align=middle width=39.53182859999999pt height=21.18721440000001pt/>. It doesn't always correct the problem, but it's worth giving it a try if you find that too many nodes are dying in your network. 

Parametric ReLU is a version of leaky ReLU where the value <img src="/tex/7e355169aef4d5008ad23b2cd4e9cf03.svg?invert_in_darkmode&sanitize=true" align=middle width=29.22385289999999pt height=21.18721440000001pt/> is replaced by a learnable parameter. 

#### Softmax

Softmax is special in the sense that it needs all of the nodes of the layer to compute the output of each node. 

<p align="center"><img src="/tex/5ef092376bd7fc00dbd85a4534138538.svg?invert_in_darkmode&sanitize=true" align=middle width=351.369876pt height=41.41941375pt/></p>

where <img src="/tex/a4c0768affcf9f52246af18c71b91211.svg?invert_in_darkmode&sanitize=true" align=middle width=16.52537039999999pt height=14.15524440000002pt/> is the input of the node being activated and <img src="/tex/4e5ed26721319d0ef4412fbd9b721a9b.svg?invert_in_darkmode&sanitize=true" align=middle width=15.17582549999999pt height=14.15524440000002pt/> is the input of every node in the same layer. 

The function always outputs a number in the range <img src="/tex/e88c070a4a52572ef1d5792a341c0900.svg?invert_in_darkmode&sanitize=true" align=middle width=32.87674994999999pt height=24.65753399999998pt/>, representing a *probability distribution*. This means that the sum of the outpus of each node of a layer will always be equal to <img src="/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>. In other words, softmax transforms any input vector of arbitrarily large or small numbers into a probability distribution. 

This is why it's often used on the output layer, where we need to classify our data into categories. 

### More error functions

#### Cross entropy

##### What is entropy?

Entropy is the average bits of transmitted information where one bit reduces our uncertainty of a situation by a factor of two. 

To be more concrete, let's illustrate it with a few examples. Say there is a match of soccer between two teams that have an equal chance of winning (50% each). If someone tells us that team 1 will win, they will have transmitted 1 bit of information to us. 

If there are 4 teams instead that are equally likely to win and someone tells us that team 1 will win, then they would have transmitted 2 bits of information because they would have reduced our uncertainty by a factor of 4 (<img src="/tex/c9fadd243c9e21b8d452d1a48b248b42.svg?invert_in_darkmode&sanitize=true" align=middle width=45.730508999999984pt height=26.76175259999998pt/>). Therefore, we can calculate the number of bits transmitted by computing the base two log of 4 (<img src="/tex/4452195ef62e364d9ce375405e28c596.svg?invert_in_darkmode&sanitize=true" align=middle width=79.55292014999999pt height=24.65753399999998pt/>). 

Now let's say that we have two teams again, but team 1 has a 75% chance of winning and team 2 has a 25% chance. If we are told that team 2 will win, then the transmitted information is <img src="/tex/2888dcda36b1f4e665448de5accbc592.svg?invert_in_darkmode&sanitize=true" align=middle width=113.34299624999998pt height=24.65753399999998pt/> bits. The uncertainty reduction is the inverse of the event's probability. Similarily, if we are told that team 1 will win, then we are given <img src="/tex/e57d75bb57d5e1ac48d5485ff98b2e35.svg?invert_in_darkmode&sanitize=true" align=middle width=134.34763979999997pt height=24.65753399999998pt/> bits. If we sum these numbers it will give us the average transmitted bits: <img src="/tex/1ed812c74e49caeac2ce4af60f146ae6.svg?invert_in_darkmode&sanitize=true" align=middle width=198.1733754pt height=21.18721440000001pt/> bits. This is called entropy and its general equation is:

<p align="center"><img src="/tex/d2eef3fe14eed090b718974fe86eac81.svg?invert_in_darkmode&sanitize=true" align=middle width=168.66628845pt height=36.6554298pt/></p>

...

<p align="center"><img src="/tex/f97a0b84890ec393511f3e1f52789ce3.svg?invert_in_darkmode&sanitize=true" align=middle width=125.6620926pt height=50.04352485pt/></p>

<p align="center"><img src="/tex/1d79ab18f324098ef3ebfc2ee070f15c.svg?invert_in_darkmode&sanitize=true" align=middle width=198.03522585pt height=50.04352485pt/></p>

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

### Optimization techniques

#### Momentum

Gradient descent is a rather dumb and shortsighted algorithm because it essentially just goes in the direction of the *current* sample of inputs, not being aware of anything else. But the current direction may not be optimal, given the curvature of the function. Gradient descent may end up traversing long shallow planes as well as steep cliffs, using the same learning rate for all terrains of the function. 

This is where momentum comes in. The momentum algorithm takes into consideration the previous gradients and influences the learning rate based on the topology of the function. If the recent gradients are all well aligned in the same direction, then the learning rate will be larger, otherwise it will be smaller. 

Recall the formula for updating weights: <img src="/tex/f29b0d74d90bf7887618e43ef4b75fe7.svg?invert_in_darkmode&sanitize=true" align=middle width=106.78507784999998pt height=28.92634470000001pt/>

We are going to introduce the momentum parameter <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/> and the current velocity <img src="/tex/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55786029999999pt height=14.15524440000002pt/> to the equation:

<p align="center"><img src="/tex/0e42a637eb2a13f4e4fefe66b329e501.svg?invert_in_darkmode&sanitize=true" align=middle width=113.61462914999998pt height=33.81208709999999pt/></p>

Therefore the first velocity value will be: <img src="/tex/17f555ab3275da0b06939609376a93d4.svg?invert_in_darkmode&sanitize=true" align=middle width=77.61472124999999pt height=28.92634470000001pt/>

The second one will take into account the first one: <img src="/tex/56543765545ec84ba46031057f6f3f98.svg?invert_in_darkmode&sanitize=true" align=middle width=122.47229939999998pt height=28.92634470000001pt/>

The third one will use the second one and so on: <img src="/tex/0f723278c9253c1dad47167fa94a7396.svg?invert_in_darkmode&sanitize=true" align=middle width=122.47229939999998pt height=28.92634470000001pt/>

#### Adam

Adam is derived from *adaptive momentum estimation* and unlike the momentum described above, it calculates a custom learning rate for each parameter. 

This is accomplished by calculating two different moving averages over the gradient. The first one is the mean and the second one is the uncentered variance. We do this by introducing two new hyperparameters (beta1 and beta2):

<p align="center"><img src="/tex/85d4959741057fd8d6c391b1adb70ac8.svg?invert_in_darkmode&sanitize=true" align=middle width=186.52399425pt height=16.438356pt/></p>

<p align="center"><img src="/tex/5d2e72c17df7d9efbf0f98e027144240.svg?invert_in_darkmode&sanitize=true" align=middle width=175.77045585pt height=18.312383099999998pt/></p>

where <img src="/tex/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode&sanitize=true" align=middle width=8.430376349999989pt height=14.15524440000002pt/> is the gradient <img src="/tex/0ede0df9c4109336d3f692690454beab.svg?invert_in_darkmode&sanitize=true" align=middle width=33.68217269999999pt height=28.92634470000001pt/>. 
The beta parameters are normally assigned values <img src="/tex/1c22e0ed21fd53f1f1d04d22d5d21677.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=21.18721440000001pt/> and <img src="/tex/a53a375441275f24641fc239deb138cb.svg?invert_in_darkmode&sanitize=true" align=middle width=37.44306224999999pt height=21.18721440000001pt/> respectively and are almost *never* changed. 

The only problem is, we need to perform bias correction since initially <img src="/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/> and <img src="/tex/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55786029999999pt height=14.15524440000002pt/> are each a vector of zeros. 

<p align="center"><img src="/tex/76e64091b792a28414e515724fd599e8.svg?invert_in_darkmode&sanitize=true" align=middle width=190.13354579999998pt height=33.85762545pt/></p>

And finally we use the moving averages to compute an individual learning rate for each parameter:

<p align="center"><img src="/tex/fef952f0d04d4c2a485a5c74a3ef03f6.svg?invert_in_darkmode&sanitize=true" align=middle width=125.59671629999998pt height=37.8236826pt/></p>

<img src="/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> being the a very small value to avoid division by zero, normally assigned <img src="/tex/3638e50463da9c60c05d8519c65d3982.svg?invert_in_darkmode&sanitize=true" align=middle width=33.26498669999999pt height=26.76175259999998pt/>. 

There are several other optimization functions, however Adam is one the most widely used and successful. 

Find the implementation in `backpropagation.go`. 

### Better weight initialization

So far we've initialized the weights as Gaussian random variables with a mean of 0 and standard deviation of 1, but we can do better than that. 

One issue with this approach is that it can cause the vanishing gradient problem. Imagine that you have 1000 input nodes all connecting to a node <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/> in the next hidden layer. If half of the inputs are 0 and half are 1, then the weighted sum <img src="/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> of node <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/> is also a Gaussian distribution with mean 0 and standard deviation <img src="/tex/6538d0c8a8bc2e68dae5cf07590e8622.svg?invert_in_darkmode&sanitize=true" align=middle width=97.7169732pt height=28.511366399999982pt/>. 

This means that <img src="/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> will often be either very large or very small. Once node <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/> is activated, it's output from a sigmoid function for example will be very close to either 1 or 0. And as we've seen before, this is the recipe for the vanishing gradient problem. 

To avoid this, we can initialize the weights of a certain layer as Gaussian random variables with mean 0 and standard deviation <img src="/tex/24b5723afa7e43a8a3487cafd30d4312.svg?invert_in_darkmode&sanitize=true" align=middle width=40.00396784999999pt height=24.995338500000003pt/> where <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> represents the number of nodes in the layer. Now if we repeat the experiment and have 500 inputs as 0 and 500 inputs as 1, the weighted sum of node <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/> will have a Gaussian distribution with mean 0, but a standard deviation of <img src="/tex/11fb6195d10a9883c1adf5f31c1a40a3.svg?invert_in_darkmode&sanitize=true" align=middle width=92.23747334999999pt height=29.424786600000015pt/> and this will avoid activating the node near it's extremes. 

Although this is relevent for weights, biases on the other hand don't benefit much from this and we can continue initializing them to zero, letting gradient descent tune them. 

Find the implementation in `layer.go`. 

## Code example

🚧UNDER CONSTRUCTION🚧

## References

- Artificial intelligence engines by James V Stone (2019)
- Complete guide on deep learning: http://neuralnetworksanddeeplearning.com/chap2.html
- Activation functions: https://medium.com/@prateekvishnu/activation-functions-in-neural-networks-bf5c542d5fec
- Mini-batch gradient descent: https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
- Batch normalization: https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c
- Momentum: https://gluon.mxnet.io/chapter06_optimization/momentum-scratch.html
- Adam: https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c