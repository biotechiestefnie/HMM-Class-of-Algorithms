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

The Viterbi path and Viterbi algorithm have become standard terms for the application of dynamic programming algorithms to maximization problems involving probabilities. Given a hidden Markov model with a set of hidden states S, a set of possible emissions (observations) M, and a sequence of T observations o0,o1,…,oT−1, the Viterbi algorithm finds the most likely sequence of hidden states that could have produced those observations. At each time step t, the algorithm solves the subproblem where only the observations up to ot are considered.

Two matrices of size T×|S| are constructed:
Pt,s contains the maximum probability of ending up at state s at observation t, out of all possible sequences of states leading up to it. Qt,s tracks the previous state that was used before s in this maximum probability state sequence. Let πs and ar,s be the initial and transition probabilities respectively, and let bs,o be the probability of observing o at state s. Then the values of P are given by the recurrence relation:

P(t,s) ={ πs x b(s,ot)  IF t = 0
          max(rinsS) (P(t-1,r) x a(r,s) x b(s,o(t))  IF t > 0 

The formula for Qt,s is identical for t >0, except that max is replaced with argmax and Q0,s=0. The Viterbi path can be found by selecting the maximum of P at the final timestep, and following Q in reverse.

**PSEUDOCODE**

```
function Viterbi(states, init, trans, emit, obs) is
    input states: S hidden states
    input init: initial probabilities of each state
    input trans: S × S transition matrix
    input emit: S × M emission matrix
    input obs: sequence of T observations

    prob ← T × S matrix of zeroes
    prev ← empty T × S matrix
    for each state s in states do
        prob[0][s] = init[s] * emit[s][obs[0]]

    for t = 1 to T - 1 inclusive do // t = 0 has been dealt with already
        for each state s in states do
            for each state r in states do
                new_prob ← prob[t - 1][r] * trans[r][s] * emit[s][obs[t]]
                if new_prob > prob[t][s] then
                    prob[t][s] ← new_prob
                    prev[t][s] ← r

    path ← empty array of length T
    path[T - 1] ← the state s with maximum prob[T - 1][s]
    for t = T - 2 to 0 inclusive do
        path[t] ← prev[t + 1][path[t + 1]]

    return path
end

```

The time complexity of the Viterbi algorithm is O(T x |S|^2)


### Forward Algorithm

The Forward algorithm, in the context of HMM, is used to calculate the probability of a state at a certain time, given the history of evidence. The process is also known as filtering. The Forward algorithm works the same way as the Viterbi algorithm except we are summing probabilities instead of taking the maximum. The main observation to take away from these algorithms is how to organize Bayesian updates and inference to be computationally efficient in the context of directed graphs of variables. The goal of the forward algorithm is to compute the joint probability (xt, y(1:t)). Once the joint probability is computed, the transition and emission probabilities are easily obtained. Both the state and observation are discrete, finite variables. The HMM's state transition probabilities, observation/emission probabilities, and initial prior probability are known, and the sequence of observations is given. Computing the joint probability would be an intractable problem, as the state sequences grow exponentially with t, so the forward algorithm takes advantage of the conditional independence rules of the hidden Markov model (HMM) to perform the calculation recursively. Forward algorithm uses the conditional indepencence of the sequence steps to calculate partial probabilities.

**PSEUDOCODE**

```
Initialize t = 0,
transition probabilities, p(xt|xt−1),
emission probabilities, p(yt|xt),
observed sequence, y(1:T)
prior probability, α(x0)

create the forward probability table 

For t = 1 to T
   α(xt) = p(yt|xt) ∑xt−1 p(xt|xt−1)α(xt−1)

Return p(xT|y1:T) = α(xT) / ∑xT α(xT)
```

The complexity of the forward algorithm is O(nm^2), where m is the number of possible states for a latent variable (like the number of weather conditions, and n is the length of the observed sequence. This is a clear reduction from the ad hoc method of exploring all the possible states, which has a complexity O(nm^n)


### Backward Algorithm

The Backward algorithm is -exactly- the same as the Forward algorithm, except you begin at the end of the sequence (including the end state) and work your way to the front. In order to calculate the probability that a position be assigned a particular state we need to understand the probability of state transitions from both directions. As a result we need to calculate probability from the reverse direction of the sequence. 












