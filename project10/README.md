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


## Baum-Welch Pseudocode

```
FUNCTION BaumWelch(observation, n_iter):

    # validate_observation raises:
    #   ValueError("Observation contains invalid characters.")
    #   ValueError("Observation contains characters not in valid_chars.")
    #   ValueError("Observation sequence is empty.")
    observation ← validate_observation(observation)

    T ← length(observation)
    states ← list of states
    symbols ← sorted list of valid emission symbols
    num_states ← number of states

    new_initial_probs ← None
    new_transition_probs ← None
    new_emission_probs ← None

    FOR iteration FROM 1 TO n_iter:

        # E-step: forward probabilities
        # forward_algorithm raises:
        #   KeyError("State not found in transition matrix.")
        #   KeyError("Symbol not found in emission matrix.")
        #   ValueError("Transition probabilities missing for state.")
        #   ValueError("Emission probabilities missing for state.")
        fwd_matrix, total_log_prob ← ForwardAlgorithm(observation)

        # E-step: backward probabilities
        # backward_algorithm raises the same errors as forward_algorithm
        bwd_matrix ← BackwardAlgorithm(observation)

        # E-step: compute gamma
        CREATE log_gamma[num_states, T]
        FOR each time t:
            log_norm_t ← logsumexp over i of (fwd[i,t] + bwd[i,t])
            FOR each state i:
                log_gamma[i,t] ← fwd[i,t] + bwd[i,t] − log_norm_t

        # E-step: compute xi
        CREATE log_xi[num_states, num_states, T−1]
        FOR each time t from 0 to T−2:
            next_symbol ← observation[t+1]

            # emission lookup raises:
            #   KeyError(f"Emission probability missing for symbol '{next_symbol}' in state '{state}'.")
            emit_next[j] ← emission log-prob of next_symbol at state j

            FOR each state i:
                FOR each state j:
                    log_xi[i,j,t] ← fwd[i,t]
                                     + transition_log_prob(i→j)
                                     + emit_next[j]
                                     + bwd[j,t+1]

            log_norm_t ← logsumexp of all log_xi[:,:,t]
            log_xi[:,:,t] ← log_xi[:,:,t] − log_norm_t

        # M-step: update initial probabilities
        new_initial_probs ← empty dictionary
        log_gamma_t0 ← log_gamma[:,0]
        log_norm ← logsumexp(log_gamma_t0)
        FOR each state i:
            normalized ← log_gamma_t0[i] − log_norm
            IF normalized is −∞: normalized ← log(1e−12)  # Incl pseudo to prevent zero probs
            new_initial_probs[state_i] ← exp(normalized)

        # M-step: update transition probabilities
        new_transition_probs ← empty nested dictionary
        FOR each state i:
            log_denom ← logsumexp(log_gamma[i,t] for t = 0..T−2)
            FOR each state j:
                log_numer ← logsumexp(log_xi[i,j,t] for t = 0..T−2)
                IF log_numer is −∞: log_numer ← log(1e−12)  # Incl pseudo to prevent zero probs
                new_transition_probs[i][j] ← exp(log_numer − log_denom)

        # M-step: update emission probabilities
        new_emission_probs ← empty nested dictionary
        FOR each state i:
            log_denom ← logsumexp(log_gamma[i,t] for all t)
            FOR each symbol s in symbols:
                mask ← all t where observation[t] = s
                IF mask not empty:
                    log_numer ← logsumexp(log_gamma[i,t] for t in mask)
                ELSE:
                    log_numer ← log(1e−12)  # Incl pseudo to prevent zero probs
                new_emission_probs[i][s] ← exp(log_numer − log_denom)

        # StrMatrix raises:
        #   ValueError("Transition matrix row missing for state.")
        #   ValueError("Emission matrix row missing for state.")
        #   KeyError("Invalid state or symbol in probability dictionary.")
        initial_probs ← new_initial_probs
        transition_probs ← new_transition_probs
        emission_probs ← new_emission_probs

        trans_matrix ← StrMatrix(transition_probs)
        emit_matrix ← StrMatrix(emission_probs)

    RETURN new_initial_probs, new_transition_probs, new_emission_probs

```

## Successes

One of the clearest successes of this week’s work was discovering that the structure we built in earlier weeks made it straightforward to extend the model with a full Baum–Welch implementation. Even though Baum–Welch itself was brand‑new code, the underlying architecture of our HMMModel class—clean separation of responsibilities, consistent internal representations, and a well‑designed interface for forward, backward, and Viterbi computations—meant that the new algorithm could be integrated without restructuring the class. This confirmed that the design principles we applied earlier in the project were effective: the model was flexible enough to support a major algorithmic addition without requiring major refactoring.

Another important success came from incorporating feedback from our peer reviewers. Their comments helped us identify places where our logic could be tightened, where numerical stability could be improved, and where our internal data structures could be made more consistent. Addressing these suggestions strengthened the reliability of our implementation and reduced the likelihood of silent errors or edge‑case failures. The end result is a more robust and maintainable model than we would have produced in isolation.

Finally, a key conceptual success was recognizing that meaningful HMM training requires treating multiple observation sequences as independent evidence rather than training on each one in isolation or concatenating them into a single artificial sequence. This insight was essential for avoiding the overconvergence and model collapse we observed during single‑sequence training, especially given the extremely short sequences in our dataset. Implementing and validating multi‑sequence Baum–Welch allowed us to correctly interpret the behavior of the model and understand why the original and multi‑trained versions produced identical log‑likelihoods. 


## Challenges
One of the most significant challenges we encountered was determining how many Baum–Welch iterations were appropriate to run the example dataset given in the Canvas instructions. We initially assumed we might need hundreds to thousands of EM iterations, but we soon learned that this is more typical for HMM training with long genomic sequences. Our example dataset, however, consisted of three rather short observation sequences (10 bp). Running 1000 iterations on such small observations caused the algorithm to over‑converge almost immediately, collapsing the probability distributions and resulting in error messages informing us that  despite using pseudocounts to prevent zero probabilities, we were dividing by zero with our total probabilities, a sure sign that all probability mass for at least one state or transition had been completely depleted, resulting in extreme overconvergence.

Because Baum–Welch repeatedly reinforces whatever patterns it detects, short sequences provide very little statistical diversity. With too many iterations, the model “memorizes” the tiny dataset instead of estimating generalizable parameters. This led us to progressively reduce the iteration count: first to 100, then to 50, and ultimately to 25 iterations, which produced stable updates without zero‑probability errors. However, it was at this point in our implementation that we realized we might also not be getting the results we were expecting because we were not technically training our model with all three observations. We had inadvertently been running the trained model on each observation sequence individually and comparing the resulting Viterbi paths, forward–backward matrices, and log‑likelihoods to those produced by the original, untrained model for each individual sequence. Baum–Welch is an expectation–maximization algorithm that reinforces whatever patterns appear in the training data, and when the data consists of only a few characters, the algorithm overfits almost immediately. This caused the model to collapse into degenerate parameter configurations, often eliminating one of the states entirely and producing dramatically worse log‑likelihoods. For example, training on OBS1 alone collapsed into an all-L state path with a log‑likelihood of –43.31, compared to the original model’s –12.48. This demonstrated severe overconvergence and loss of meaningful structure. OBS2 and OBS3 produced log likelihoods around -11.09, with each pulling the model in different directions.

Because single‑sequence training was not providing useful insight into the accuracy or stability of the optimal paths, we implemented a multi‑sequence Baum–Welch wrapper in the implementation notebook that accumulates expected counts (gamma and xi) across all three sequences before performing each M‑step update. This is the statistically correct way to train an HMM on multiple independent observations, and we included this outside of our main script to preserve the stable, reusable logic of our HMMModel, since this was an experimental decision and not a universal HMM feature. Importantly, we also confirmed that simply concatenating the sequences into one long observation is not valid: doing so forces the model to treat artificial boundaries as real transitions, distorting the initial state distribution and producing parameter updates that do not reflect the true structure of the data. Multi‑sequence EM avoids these issues by treating each sequence independently during the E‑step while still learning shared parameters during the M‑step.

After implementing multi‑sequence training, we compared the trained model back to the original model. The results of training the model with all three sequences rather than one caused the log‑likelihoods for all three sequences to become VIRTUALLY IDENTICAL between the original model and the multi‑sequence trained model:


  * OBS1: –12.483820056859475

  * OBS2: –12.483043753806445

  * OBS3: –12.490497134887951


The Viterbi paths and posterior state sequences were also identical. This indicates that the original model was already at a local optimum for this extremely small dataset, and the Baum–Welch updates produced no meaningful parameter changes. In other words, the dataset was too short and too limited to shift the model away from its initial configuration, so the multi‑sequence EM converged immediately to the overall optimal path, but was able to escape overconverging.


Overall, the multi‑sequence training experiment demonstrated that:

* Single‑sequence Baum–Welch is not informative for extremely short sequences due to overfitting and parameter collapse.

* Concatenation is not a valid alternative, as it introduces artificial transitions and distorts the statistical structure of the data.

* Multi‑sequence EM is the correct approach, and in our case, it revealed that the original model was already at a stable optimum given the limited dataset.

Identical log‑likelihoods between the original and multi‑trained models confirm that no parameter drift occurred, while the large discrepancies in the single‑sequence trained models illustrate the dangers of training on insufficient data- each tiny sequence exerts disproportionate influence, leading to unstable and biologically meaningless parameter updates.

Taken together, these experiments demonstrate why multi‑sequence Baum–Welch is essential for meaningful HMM training on small biological datasets. Training on individual sequences produced unstable, contradictory, and biologically implausible models, while concatenating sequences would have introduced artificial transitions and distorted the underlying statistical structure. By instead applying the correct multi‑sequence EM procedure, we were able to evaluate whether the model could learn consistent parameters from all three observations without compromising their independence. Furthermore, adding this modification inside the implementation notebook allowed us to keep our HMMModel logic stable and reusable. The fact that the multi‑sequence trained model produced identical log‑likelihoods, Viterbi paths, and posterior state assignments to the original model indicated that the initial parameters were already at a local optimum for this limited dataset, and that the available evidence was insufficient to shift the model in any meaningful direction. In contrast, the dramatic divergence observed in the single‑sequence trained models highlights the dangers of overfitting when data are sparse. Ultimately, this comparison reinforces both the importance of proper multi‑sequence training and the limitations imposed by extremely short sequences, while confirming that the original model remained the most stable and reliable representation of the underlying process given the data available, and it was an invaluable lesson for us.


## Reflections

### Group Leader: Stefanie Moreno


### Collaborators: Thu Thu Han
This week's Baum-Welch algorithm was more challenging for me to grasp compared to Viterbi and Forward-Backward, partly because I missed class and could not spend enough time to wrap my head around the four major calculations required for this algorithm due to my other assignments this week. The part I found hardest was understanding how the four calculations forward, backward, gamma, and xi connect and build on each other to re-estimate the model parameters. Thankfully my teammates were on top of the algorithm this week and were very helpful in bridging that gap. They explained each step relating back to last week’s forward and backward algorithm and some issues we came across. Going through the pseudocode step by step and seeing how gamma and xi feed into re-estimating the initial, transition, and emission probabilities really helped me see the algorithm as a whole rather than isolated steps. 

### Collaborators: Eric Arnold




### Generative AI Appendix
Generative AI assistance was saught this week using Anthropic's Claude to answer some questions we had regarding the input examples and minor revisions to our HMMModels class to ensure robust implementation of Baum-Welch

Prompts:
We currently have a Baum–Welch implementation that only accepts one observation at a time. We want to compare the results of our original model to the trained model using the original model's initial parameters. This includes 3 observation sequences (~10bp each), and initial, transmission, and emission probabilities. We will be using the original model's probabilities as informed initialization for the trained model's initial parameters.
1. How should we modify Baum–Welch so that it trains from all 3 sequences?
2. What should we compare the trained model to? The original model was never trained on all 3 sequences, so should the comparison be:
   - original model likelihood on each sequence,
   - original model likelihood summed across all sequences,
   - or something else?
Please explain the correct conceptual workflow and provide examples of how we could execute multipe observations at once

The following two scripts define our class implementation of HMMModel. Upon peer review, we were informed that one of the calculations in the forward backward algorithm was incorrectly written. Can you identify this and show us why it was wrong conceptually given the design considerations of our code, and also provide the correct calculation along with a detailed explanation of each variable in the calculation and why it is calculated in this manner.

Given the two scripts defining our HMMModel, are the rows and columns being accessed properly? We are not getting rows that are adding to 1 and we are trying to troubleshoot to identify if something was miswritten.

Regarding the Baum-Welchh algorithm, do the initial parameters have to come from a randomized selection? Does it matter what values are used as the intial parameters? How does using informed initialization compare to randomized?
