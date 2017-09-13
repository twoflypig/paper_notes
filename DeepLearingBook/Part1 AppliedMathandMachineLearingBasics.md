# 2.8 Singular Value Decomposition

for example, if a matrix is not square, the eigendecomposition is not defined, and we must use a singular
value decomposition instead 

We can actually interpret the singular value decomposition of A in terms of the eigendecomposition of functions of A. The left-singular vectors of A are the eigenvectors of AA. The right-singular vectors of A are the eigenvectors of A A.
The non-zero singular values of A are the square roots of the eigenvalues of A A. The same is true for AA  

# 2.9 The Moore-Penrose Pseudoinverse

![1](1.PNG)

![2](2.PNG)

# 2.10 The Trace Operator

The trace of as square martix composed of many factors is also invariant to moving the last factor into the first position.please see the book

# 2.11 The Determinant

The product is equal to the product of all the eigenvalues of the martix

# 2.12 Example :Principal Components Analysis

TODO:

# 3 Probability and Information Theory

First, the laws of probability tell us how AI systems should reason, so we design our algorithms to compute or approximate various expressions derived using probability theory. Second, we can use probability and statistics to theoretically analyze the behavior of proposed AI systems. (Design and analyze)

# 3.1 Why Probability

Bayesian probability and frequentist probability

# 3.3 Probability

## 3.3.1 Discrete Variables and Probability Mass Functions

![3](3.PNG)

![4](4.PNG)

![5](5.PNG)

# 3.9 Common Probability Distributions

## 3.9.6 Mixtures of Distributions

**Latent variables**

![6](6.PNG)

![7](7.PNG)

# 3.12 Technical Details of Continuous Variables

# 10 Sequence Modeling:Recurrent and Recursive Nets

The time step index need not literally refer to the passage of time in the real wordld, but only to the **position ** in the sequence. RNNs can also by applied in two dimensions across spatial data such as images, and even when applied to data involving time.

## 10.1 Unfolding Computational Graphs

$h_t = f(h_{t-1},x_t;$$\theta$)

To indicate that the state is the hidden units of the network .  using variable $h$ to represent the state. Typical RNNs will add extra architectural features such as output layers that read information out of the state $h$ to make predictions

We can view $h^t$ as a kind of lossy **summary **of the task-relevent aspects of the past sequence of inputs up to t , sine it maps an arbitrary length sequence ($x^t$,$x^{t-1}$,....) to a fixed length vector $h^t$ . For example , we ask $h^t$ to be rich enough when we want to predict the rest of the sentence.

The unfolding process thus introduces two major advantages:

- Regardless of the sequence length . 
- it is possible to use the same transition function f with the same parameters at every time step.

Learning a single ,shared model allows **generalization** to sequence lengths that did not appear in the traning set, and allows the model to be estimated with far fewer training examples than would be required without parameter sharing.

**powerful but also expensive to train**

## 10.2 Teacher Forcing and Networks with Output Recurrence

The network lacks hidden-to-hidden recurrence **requires that the output units capture all of of the information about the past that the network will use to predict the future**. Because the output units are explicitly trained to match the training set targets ,they are unlikely to capture the necessary information to match the training set targets, they are unlikely to capture the necessary information about the past history of the input, unless the user knows how to discribe the full state of the system and provides it as part of training set targets.

![8](8.PNG)

As soon as the hidden units become a function of earlier time steps,the BPTT algorithm is necessary. Some models may thus be trained with both teacher forcing and BPTT.

You can randomly select the output of model or the actual data values as input.

## 10.2.2 Computing the Gradient in a Recurrent Neural Network







