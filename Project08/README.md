# Introduction
Description of the project

# Pseudocode
Put pseudocode in this box:

```
Needs:                                                                                    Vocabulary class
Beta- starting probabilities                                            alphabet
Final probabilities will be $
States 
Transition probabilities
Emission probabilities

pseudocode
Initialize 2 matrices
Using beta, compute first column states
For column in remaining columns:
    Let i be the row
    Let j be the column
    For each cell:
        Take max of cross transition probabilities
(1st calculation (j-1) x transition x emission  2nd calculation (i +1 ) (j-1) x transition x emission and take max of the calculations)
        Multiply max prob by emission to get state
        Save argmax as best_path
```

# Successes
Description of the team's learning points

# Struggles
Struggles:
 
We didn't program this recursively because we were struggling to understand the algorithm as-is. However, were we to take a recursive approach, each step forward in the matrix would involve a function call. We could save the best state associated with the scores in a 1-D state array and call the function again with the indices incremented forward. Upon reaching the end, we would return the best state, and this would give rise to the traceback that could be carried back through the recursive layers.
 
A key issue for us was in figuring out how traceback worked. Initially, we presumed that we could just take the max score of the viterbi matrix. Upon closer inspection, this would have just given us a naive greedy algorithm. It is only when we know which preceding states are optimal that we can determine which current state is optimal. This is why the algorithm has two distinct parts. Using the next state to predict the previous state was a tricky thing to figure out logically.


# Personal Reflections
## Group Leader
Group leader's reflection on the project

## Other members
Eric's reflection:
 
As with smith-waterman, it was interesting to see how the algorithm used subproblems to solve the overall problem. Essentially the number of possible states given the emissions is a huge tree that is pruned at every forward iteration. Once we reach the end, this fixes the best probability and determines which "path" we should take back through the assigned states. Like BWT, it was a fun algorithm to work on together because there were a few twists and turns that caused us to question our thinking.
 

# Generative AI Appendix
As per the syllabus
