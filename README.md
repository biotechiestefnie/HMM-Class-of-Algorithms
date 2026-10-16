# Hidden Markov Models

Hidden Markov Models Virtibi, Forward, Backward, and Forward-Backwards Algorithms, Baum-Welch Profile HMM Implementation


## Objective

This collaborative project spanned four weeks and entailed the demonstration of five core dynamic programming methods used to analyze and train Hidden Markov Models (HMMs). In probability theory, an HMM is a Markov model in which the observations are dependent on a latent Markov process (referred to as X). HMMs are are a surprisingly powerful tool for modeling a wide range of sequential data, making them well-suited for genomic sequence analysis. The hidden and observable states of an HMM can be defined as sequences of random variables, meaning it is stochastic.


## Background: Components of the Hidden Markov Model

* Hidden States: The actual conditions or causes driving the system, which cannot directly be seen. The hidden states collectively form a Markov Chain, as in, chaining together multiple hidden states that are traversed over time in order to reach an outcome. This is a probabilistic process because all the parameters of the Markov Chain, as well as the score of each sequence, are in fact probabilities.

* Observations: The visible data or outputs that can be measured. 

* Markov Assumption: The rule that the next hidden state depends only on the current state, not on any past history. Therefore, each observation is only dependent on the state that produced it, and is completely independent from any other state in the chain.

* Transition Probabilities: The odds of moving from one hidden state to another.

* Transition Matrix: The probability of going from one state to another is captured in a Transition Matrix. This matrix must also be row stochastic meaning that the probabilities from one state to any other state in the chain (each row in the matrix) must sum to one.

* Emission Probabilities: The odds of seeing a specific observation while in a particular hidden state.

* Initial Probability Distribution: A vector (π) that defines the probability of the system starting in each hidden state at time step zero or one.

* Sequence of Observations: An ordered chain of visible data points or symbols emitted by the system over discrete time steps.

* Observation Likelihood Matrix: A table of probabilities that shows how likely each visible output or symbol is to be generated from a specific hidden state. It is also commonly called the emission probability matrix or state-observation matrix.


### Vetirbi Algorithm

The Viterbi path and Viterbi algorithm have become standard terms for the application of dynamic programming algorithms to maximization problems involving probabilities. Given a hidden Markov model with a set of hidden states S, a set of possible emissions (observations) M, and a sequence of T observations o0,o1,…,oT−1, the Viterbi algorithm finds the most likely sequence of hidden states that could have produced those observations. At each time step t, the algorithm solves the subproblem where only the observations up to ot are considered.

Two matrices of size T×|S| are constructed:
Pt,s contains the maximum probability of ending up at state s at observation t, out of all possible sequences of states leading up to it. Qt,s tracks the previous state that was used before s in this maximum probability state sequence. Let πs and ar,s be the initial and transition probabilities respectively, and let bs,o be the probability of observing o at state s. Then the values of P are given by the recurrence relation:

P(t,s) ={ πs x b(s,ot)  IF t = 0
          max(rinsS) (P(t-1,r) x a(r,s) x b(s,o(t))  IF t > 0 

The formula for Qt,s is identical for t >0, except that max is replaced with argmax and Q0,s=0. The Viterbi path can be found by selecting the maximum of P at the final timestep, and following Q in reverse.


**PSEUDOCODE**

```

CLASS HMMModel:

    METHOD __init__(initial_probabilities, transition_probabilities, emission_probabilities):
        Store the initial, transition, and emission probability dictionaries
        Extract list of states
        Convert all probabilities to natural log values
        Replace any zero probabilities with negative infinity

    METHOD viterbi_algorithm(observation_sequence):

        Determine the number of states and the length of the observation sequence

        Create a Viterbi matrix with one row per state and one column per observation
        Create a traceback matrix with the same dimensions

        For the first observation:
            For each state:
                Retrieve the initial probability for that state
                Retrieve the emission probability for first observation character
                Convert both to log values
                Add them (because using natural log) and store result in first column of Viterbi matrix
                Store placeholder in traceback matrix

        For each remaining observation position:
            For each current state:
                Retrieve the emission probability for the current symbol and convert to ln
                For each possible previous state:
                    Retrieve previous Viterbi score
                    Retrieve transition probability from previous state to current state
                    Convert transition probability to ln
                    Compute: previous score + log transition + log emission
                Identify previous state yielding maximum total
                Store this maximum in Viterbi matrix
                Store index of maximizing previous state in traceback matrix

        Identify state with highest log‑probability in final column

        Create empty list for optimal path
        Append final state

        Move backward through traceback matrix:
            At each column, use stored previous‑state index to determine state that led here
            Append that state to path
            Continue until reaching first column

        Reverse path to run from first observation to last

        Return Viterbi matrix, traceback matrix, and optimal state path.


```

The time complexity of the Viterbi algorithm is O(T x |S|^2)


### Forward Algorithm

The Forward algorithm, in the context of HMM, is used to calculate the probability of a state at a certain time, given the history of evidence. The process is also known as filtering. The Forward algorithm works the same way as the Viterbi algorithm except we are summing probabilities instead of taking the maximum. The main observation to take away from these algorithms is how to organize Bayesian updates and inference to be computationally efficient in the context of directed graphs of variables. The goal of the forward algorithm is to compute the joint probability (xt, y(1:t)). Once the joint probability is computed, the transition and emission probabilities are easily obtained. Both the state and observation are discrete, finite variables. The HMM's state transition probabilities, observation/emission probabilities, and initial prior probability are known, and the sequence of observations is given. Computing the joint probability would be an intractable problem, as the state sequences grow exponentially with t, so the forward algorithm takes advantage of the conditional independence rules of the hidden Markov model (HMM) to perform the calculation recursively. Forward algorithm uses the conditional indepencence of the sequence steps to calculate partial probabilities.

**PSEUDOCODE**

```

function forward:

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

The Backward algorithm is -exactly- the same as the Forward algorithm, except calculations begin at the end of the sequence (including the end state) and work towards the front. The output of this algorithms represents the probability of being in a certain state at a certain time, given all future observations In order to calculate the probability that a position be assigned a particular state the probability of state transitions from both directions needs to be understood. As a result the probability from the reverse direction of the sequence must be claculated. 

We start by setting the backward probability of each state at the final time step to 1. The reason we initialize the backward probabilities to 1 at the final time step in the Backward Algorithm is because we define the backward probability as the probability of the future observations given the current state. At the final time step, there are no future observations because we’ve reached the end of the sequence. Therefore, the probability of observing the “empty” sequence of future observations given any current state is 1. This is a convention used to simplify the calculations and provide a base case for the recursion in the Backward Algorithm. Mathematically, this can be represented as: βi​(T)=1 for all states i, where T is the final time step.

For each preceding time step t (from T−1 to 1), we calculate the backward probability of each state i. This is done by summing over the products of the backward probabilities of each possible next state j, the transition probability from the current state i to that next state j, and the emission probability of the next observation given that next state j. Mathematically, this is represented as:

                  *Bi*(t) = ∑ (*Bj*(t + 1) x αij x *bj*(Ot+1)
                           jtoS
where: 

*Bi*(t) is the backward probability for state i at time t 

S is the set of all possible states

*Bj*(t + 1) is the backward probability of state j at time t+1

αij is the transition state i to state j

*bj*(Ot+1) is the probability of observing Ot+1 given j

For each preceding time step (from 2 to 1), the backward probability of each state is calculated. This is done by summing over the products of the backward probabilities of each possible next state, the transition probability from the current state to that next state, and the emission probability of the next observation given that next state.

The total probability of the observed sequence is calculated by summing over the products of the initial state probabilities, the emission probabilities of the first observation given each state, and the backward probabilities of each state at the first time step. This constitutes the termination, and all contributions are summed to get the total probability of the observed sequence.

**PSEUDOCODE**

```

function bwd:
    initialize zeros matrix of size n_states x observations
    reverse observations
    
    set first col to 0 (which is a probability of 1)
    
    for col in remaining cols: # same loop as the first time except everything is backwards and the start is 1
        for state in states:
            matrix[row][col] = log_sum_exp(prev_score[row][prev_col] x transition_matrix[state][row] x emission_matrix[state][observation])
    
    return matrix

```


## Forward-Backward Algorithm

The Forward-Backward algorithm combines both the forward and backward matrices to compute the posterior probability of each 
state at each position in the final forward-backward matrix. For each cell (i, j) in the forward backward matrix, the posterior 
is computed as the sum of the log- forward score and log- backward score at that position, minus the total log-probability of the 
observation sequence. The total log-probability is obtained by applying np.logaddexp.reduce to the last column of either the forward matrix or backwards matrix.

**PSEUDOCODE**

```

function fwd-bwd:

    FWD, P_seq = fwd
    BWD = bwd

    G = (FWD * BWD) / P_seq

    return states associated with argmax of each column

```





