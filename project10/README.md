# Project10: Baum-Welch Algorithm

## Introduction

Baum–Welch is a method that allows a Hidden Markov Model to learn its own 
parameters directly from data, even when the hidden states are never observed. 
An HMM has three sets of parameters: the probability of starting in each state, 
the probability of transitioning between states, and the probability of emitting 
each observable symbol from each state. When these probabilities are unknown, 
Baum–Welch repeatedly analyzes the observed sequence and adjusts the model so that 
the sequence becomes more likely under the model. In other words, it tunes the HMM 
to explain the data as well as possible.

To do this, the algorithm first uses the current version of the model to compute how 
the hidden states behave in expectation across the entire observation sequence. It does 
this by running the forward and backward algorithms, which together, provide a robust 
prediction of how likely each state is at each position, and how probable each state‑to‑state 
transition is between positions. These quantities are not guesses; they are mathematically 
derived expectations based on the model’s current parameters. They represent the model’s best 
estimate of what the hidden state sequence “must have looked like” to have produced the 
observed data.

Once these expectations are computed, Baum–Welch rebuilds the model’s parameters so that they 
match what the model predicts happened. If the model spent a large amount of time in a 
particular state at the beginning of the sequence, the initial probability for that state is 
increased. If transitions from one state to another occurred frequently, the transition  
probability between those states is increased. If a state is expected to have produced 
a particular symbol many times, the emission probability for that symbol from that state is 
increased. Every parameter is recalculated from these expected counts, producing a new version of 
the HMM that better fits the data.

This parameter evaluation process is repeated over multiple iterations. Each iteration uses the 
updated model to produce new expectations, and those expectations produce new parameters. With 
each cycle, the model becomes more consistent with the data, and the likelihood of the observed 
sequence under the model increases. Baum–Welch does not guarantee that the model reaches the 
absolute best possible parameters, but it does guarantee that each iteration improves or 
maintains the likelihood. Over time, the HMM converges to a set of parameters that explain the 
observed sequence as well as the model’s structure allows.

In short, Baum–Welch is the mechanism that lets an HMM teach itself: it infers how the hidden states must 
have behaved, uses those inferences to rebuild the model, and repeats this refinement until the model 
stabilizes. Over the past few weeks, we have continually expanded upon HMM implementations- starting 
with Vetirbi, then Forward, Backward, and Forward-Backward algorithms, to this week's addition: Baum-
Welch. We have implemented Object-Oriented Programming to ensure our program was extensible, and we
ultimately executed the code for each algorithm in a Jupyter Notebook. As such, this week's project
contains all previous implementations, displaying a progression of our learning. However, we have updated
the implementation sequences, and initial, transmission, and emission probabilities to reflect those
provided to us for Module 10, and we have run all algorithms with this input data.