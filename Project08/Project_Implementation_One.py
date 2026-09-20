import numpy as np
from tabulate import tabulate
class HMMVocab:
    """
    Container class that handles input sequence processing.
    """

    def __init__(self):
        # hardcoded valid nucleotides
        self.valid_nucleotides = {"A", "C", "G", "T"}

    def validate_observation(self, observation):
        # list composition
        # go over each nucleotide in observation and only keep nucleotides in the valid set
        cleaned = [nucleotide for nucleotide in observation if nucleotide in self.valid_nucleotides]
        return "".join(cleaned)

class HMMModel:

    def __init__(self, initial_probs, transition_probs, emission_probs):

        self.initial_probs  = initial_probs
        self.transition_probs = transition_probs
        self.emission_probs  = emission_probs
        # create a np array of all possible hidden states to map directly to matrix rows
        self.states      = np.array(list(initial_probs.keys()))
        self.vocab = HMMVocab()

    def viterbi_algorithm(self, observation):

        # validate the observation sequence
        observation = self.vocab.validate_observation(observation)
        # get number of rows in the matrix
        num_states = len(self.states)
        # get number of columns in the matrix
        num_cols = len(observation)

        # initialise viterbi matrix with header row and header column
        viterbi_matrix = np.zeros((num_states + 1, num_cols + 1), dtype=object)
        for j, nucleotide in enumerate(observation):
           viterbi_matrix[0][j + 1] = nucleotide
        for i, state in enumerate(self.states):
            viterbi_matrix[i + 1][0] = str(state)
        viterbi_matrix[0][0] = ""

        # initialise traceback matrix with same shape filled with None
        traceback_matrix = np.empty((num_states + 1, num_cols + 1), dtype=object)
        for j, nucleotide in enumerate(observation):
            traceback_matrix[0][j + 1] = nucleotide
        for i, state in enumerate(self.states):
            traceback_matrix[i + 1][0] = str(state)
        traceback_matrix[0][0] = ""

        # first observation column
        for i, state in enumerate(self.states):

            row = i + 1 # skipping the header row

            initial = self.initial_probs[state] # get initial probability for current state

            # get emission probability for this state and observation
            #col 1 = observation[0]
            #col 2 = observation[1]
            #col 3 = observation[2]
            emission = self.emission_probs[state][observation[0]]

            viterbi_matrix[row][1] = initial * emission

            traceback_matrix[row][1] = str(state)

        # loop starts
        for j in range(2, num_cols + 1): # current column

            for i, state in enumerate(self.states): # current row

                row = i + 1 #skip header row
                scores = viterbi_matrix[1:, j-1]

                transition = np.array([self.transition_probs[prev_state][state] for prev_state in self.states])

                emission = self.emission_probs[state][observation[j - 1]]

                scores = scores * transition * emission

                # Get the max scores and state it came from
                best_prob = np.max(scores)

                best_state = self.states[np.argmax(scores)]

                # Fill the matrices with scores and states
                viterbi_matrix[row][j] = best_prob

                traceback_matrix[row][j] = best_state

        # traceback
        # Get the last column from viterbi matrix skipping header
        last_column = viterbi_matrix[1:, num_cols]

        # get the last column maximum index
        last_col_max = np.argmax(last_column)

        # get the corresponding index of maximum from traceback matrix
        # + 1 to get the matrix row index
        current = traceback_matrix[last_col_max+1][num_cols]

         # add it to path as string
        path = [str(current)]

        current = traceback_matrix[last_col_max + 1][num_cols]

        # add it to path as string
        path = np.full(num_cols, 0, dtype=object)
        path[-1] = str(current)

        # start at num_col -1 as we already got the last column max
        # stop before 0
        # step -1 backwards
        state_list = list(self.states)
        for j in range(num_cols - 1, -1, -1):
            row = state_list.index(path[-1])
            col = j + 1
            path[j] = str(traceback_matrix[row, col])

        print(tabulate(viterbi_matrix, tablefmt="plain"))
        print(tabulate(traceback_matrix, tablefmt="plain"))
        print(last_column)
        print(last_col_max)
        print(current)
        print(path)

        return path


obs = "GGCAaCTNNGAA"

init_probs = {
    "I": 0.2,
    "G": 0.8
}

trans_probs = {
    "I": {"I": 0.7, "G": 0.3},
    "G": {"I": 0.1, "G": 0.9}
}

emit_probs = {
    "I": {"A": 0.1, "C": 0.4, "G": 0.4, "T": 0.1},
    "G": {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
}

model = HMMModel(init_probs, trans_probs, emit_probs)
model.viterbi_algorithm(obs)

