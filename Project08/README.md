# Introduction
This project implements a Hidden Markov Model (HMM) with the Viterbi algorithm via an object oriented programming design. The model stores initial, transition, and emission probabilities, and the Viterbi method computes the most likely hidden state sequence for a given observation sequence string. All probability calculations are performed in natural log, which transforms the exceptionally small probabilities resulting from repeated multiplication of probability values in decimal format into a sum of log‑probabilities. This conversion helps to avoid numerical underflow and creates a stable, additive dynamic programming formulation.

The class structure is intentionally minimal and extendable. Probabilities are supplied as dictionaries, making it straightforward to add new states, modify emissions, or integrate additional HMM methods without rewriting core logic. The implementation follows a consistent formatting and documentation style to keep the code readable, reproducible, and aligned with standard Python pep8 style guidelines.

The Viterbi algorithm is a dynamic programming approach that constructs a matrix of best log‑probabilities along with a corresponding traceback matrix. Each cell represents the best score of reaching a state at a given position, computed by combining the previous state’s score, the transition log‑probability, and the emission log‑probability for the current observed character. After the matrix is filled, a backward traceback reconstructs the optimal hidden path.

For an HMM with N states and an observation sequence of length T, the Viterbi algorithm runs in *O(N^2T)* time and *O(NT)* space. This complexity is acceptable for smaller state spaces and moderate sequence lengths, and the class design allows for the addition of optimization techniques if necessary.

# Pseudocode

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

# Successes
One of the clearest successes of this project was how we adapted our design upon improving our understanding of the problem. Early on, we planned to build a dedicated vocabulary class to manage symbols and indexing, but as the implementation matured, Thu Thu recognized that a simple Python set provided the same functionality with far less overhead. Making this adjustment simplified the codebase, reduced unnecessary abstraction, and kept the model focused on the core HMM logic rather than unnecessary and excessive class implementation.

We also succeeded in producing an extensible object oriented programming design for the HMM/Viterbi algorithms that yielded the expected outputs upon execution. Converting all probabilities to natural log space early in initialization made the Viterbi computation stable with more interpretable values, and separating the model definition from the driver code kept the notebook organized and reproducible. The team consistently applied style‑guide formatting, documented key decisions, and reached a mutual understanding of the algorithm. These choices made the final implementation logically sensible, easy to test, and predictably extendable, which ultimately indicates our deeply collaborative effort.

# Struggles
Struggles:
 
We didn't program this recursively because we were struggling to understand the algorithm as-is. However, were we to take a recursive approach, each step forward in the matrix would involve a function call. We could save the best state associated with the scores in a 1-D state array and call the function again with the indices incremented forward. Upon reaching the end, we would return the best state, and this would give rise to the traceback that could be carried back through the recursive layers.
 
A key issue for us was in figuring out how traceback worked. Initially, we presumed that we could just take the max score of the viterbi matrix. Upon closer inspection, this would have just given us a naive greedy algorithm. It is only when we know which preceding states are optimal that we can determine which current state is optimal. This is why the algorithm has two distinct parts. Using the next state to predict the previous state was a tricky thing to figure out logically.


# Personal Reflections
## Group Leader
Stefanie Moreno's reflection: Leading this project gave me a chance to create and manage our repository while collaborating to create a program we were proud of. One of the most valuable parts of the process was helping the team design a clean, extensible object-oriented implementation that followed Python pep8 style guidelines. We made deliberate choices such as performing all probability calculations in natural log space and keeping the Viterbi logic modular to ensure the code remained stable, readable, and extensible for the coming weeks' assignments. I also focused on maintaining consistent formatting and documentation so that each component fit together properly and we would be able to look back at our script and easily understand the logic of the code.

I am proud of how our team handled the debugging and refinement stages, especially when resolving differences in interpretation and making critical last-minute decisions. Our virtually incessant communication helped clarify the algorithm for all of us and yielded a clear and concise visualization of our results upon execution of the program with a small sample observation sequence. Overall, the project was a good balance of collaboration and problem solving, and I feel we produced a reliable implementation of the Viterbi algorithm.

## Other members
Eric's reflection:
 
As with smith-waterman, it was interesting to see how the algorithm used subproblems to solve the overall problem. Essentially the number of possible states given the emissions is a huge tree that is pruned at every forward iteration. Once we reach the end, this fixes the best probability and determines which "path" we should take back through the assigned states. Like BWT, it was a fun algorithm to work on together because there were a few twists and turns that caused us to question our thinking.
 

# Generative AI Appendix
Application: Microsoft CoPilot

Prompts:

    - Given the following code, how can we tighten up the comments while still defining logic and reasoning effectively?
    - In converting a Python script from a .py file to a Jupyter Notebook, should there be any specific additions or omissions? Is it better to place the entire script into one cell or should it be modularized in separate cells? 
    - Where should helper functions, classes, and driver code be placed in a notebook for readability, reproducibility, and proper execution?
    - What is the cleanest way to restructure our code to make the output matrices properly formatted?

Justification of Use:
AI support was used to streamline documentation and to verify that explanations were technically accurate and comprehensible. We also elicited support in formatting our output after transferring the script from a PyCharm .py file to a Jupyter Notebook because initially, the float values were not presented in scientific notation and were difficult to interpret. The long decimal values also interfered with the presentation of the optimal hidden path, making characters appear disconnected and hard to read. Based on guidance from CoPilot, we resolved this problem by restructuring the notebook, separating helper functions from the driver code, and applying a consistent formatting function to render matrices in a readable HTML table. This kept the output interpretable and aligned with our expectations for visual clarity and easily interpretable results.
