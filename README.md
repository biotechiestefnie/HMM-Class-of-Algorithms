# Hidden Markov Models

Hidden Markov Models Virtibi, Forward, Backward, and Forward-Backwards Algorithms, Baum-Welch Profile HMM Implementation


## Objective

This collaborative project spanned four weeks and entailed the demonstration of five core dynamic programming methods used to analyze and train Hidden Markov Models (HMMs). In probability theory, an HMM is a Markov model in which the observations are dependent on a latent Markov process (referred to as X). HMMs are are a surprisingly powerful tool for modeling a wide range of sequential data, making them well-suited for genomic sequence analysis. The hidden and observable states of an HMM can be defined as sequences of random variables, meaning it is stochastic.

* The Hidden States: The actual conditions or causes driving the system, which cannot directly be seen. For example, the weather might be "Sunny" or "Rainy" inside a room with no windows. The hidden states collectively form a Markov Chain, as in, chaining together multiple hidden states that are traversed over time in order to reach an outcome. This is a probabilistic process because all the parameters of the Markov Chain, as well as the score of each sequence, are in fact probabilities.

* The Observations: The visible data or outputs that can be measured. For example, whether a person carries an umbrella or not.

* The Markov Assumption: The rule that the next hidden state depends only on the current state, not on any past history. Therefore, each observation is only dependent on the state that produced it, and is completely independent from any other state in the chain.


**Core Probabilities**

* Transition Probabilities: The odds of moving from one hidden state to another (e.g., how likely a rainy day is to follow a sunny day).

* Emission Probabilities: The odds of seeing a specific observation while in a particular hidden state (e.g., how likely someone is to carry an umbrella when it is raining)


In order to build a HMM you need:

- Hidden States
- 
- Transition Matrix: The pprobability of going from one state to another is captured in a Transition Matrix. This matrix must also be row stochastic meaning that the
                     probabilities from one state to any other state in the chain (each row in the matrix) must sum to one.
  
- Sequence of Observations: An ordered chain of visible data points or symbols emitted by the system over discrete time steps

- Observation Likelihood Matrix: A table of probabilities that shows how likely each visible output or symbol is to be generated from a specific hidden state. It is also commonly called the emission probability matrix or state-observation matrix.

- Initial Probability Distribution: A vector (π) that defines the probability of the system starting in each hidden state at time step zero or one



The Viterbi algorithm is an efficient method of finding a sequence z1,...,zn with maximal probability given x1,...,xn, that is, finding z(1:n) ∈ argmax z1:n p(z1:n|x1:n).

