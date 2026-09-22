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
   
- Transition Matrix: The probability of going from one state to another is captured in a Transition Matrix. This matrix must also be row stochastic meaning that the probabilities from one state to any other state in the chain (each row in the matrix) must sum to one.
  
- Sequence of Observations: An ordered chain of visible data points or symbols emitted by the system over discrete time steps

- Observation Likelihood Matrix: A table of probabilities that shows how likely each visible output or symbol is to be generated from a specific hidden state. It is also commonly called the emission probability matrix or state-observation matrix.

- Initial Probability Distribution: A vector (π) that defines the probability of the system starting in each hidden state at time step zero or one


### Vetirbi Algorithm

The Viterbi algorithm is a dynamic programming algorithm that finds the most likely sequence of hidden events that would explain a sequence of observed events. The result of the algorithm is often called the Viterbi path. It is most commonly used with HMMs. Viterbi path and Viterbi algorithm have become standard terms for the application of dynamic programming algorithms to maximization problems involving probabilities. Given a hidden Markov model with a set of hidden states S, a set of possible emissions (observations) M, and a sequence of T observations o0,o1,…,oT−1, the Viterbi algorithm finds the most likely sequence of hidden states that could have produced those observations. At each time step t, the algorithm solves the subproblem where only the observations up to ot are considered.

Two matrices of size T×|S| are constructed:
Pt,s contains the maximum probability of ending up at state s at observation t, out of all possible sequences of states leading up to it. Qt,s tracks the previous state that was used before s in this maximum probability state sequence. Let πs and ar,s be the initial and transition probabilities respectively, and let bs,o be the probability of observing o at state s. Then the values of P are given by the recurrence relation:

    

