from IPython.display import HTML
from tabulate import tabulate
import numpy as np
from dict_maker import StrMatrix

def show_matrix_html(matrix, float_format="{:.3e}"):
    formatted = []
    for row in matrix:
        new_row = []
        for cell in row:
            if isinstance(cell, float):
                new_row.append(float_format.format(cell))
            else:
                new_row.append(str(cell))
        formatted.append(new_row)

    return HTML(tabulate(formatted, tablefmt="html"))



class HMMModel:
    """
    Hidden Markov Model for computing the most likely hidden-state sequence
    given an observation sequence, using natural log (ln) probabilities for
    numerical stability and underflow prevention.

    Attributes:
        initial_probs (dict): Probability of starting in each state.
        transition_probs (dict:dict): Transition probabilities between states.
        emission_probs (dict:dict): Emission probabilities for each state.
        states (np.ndarray): Array of hidden states for matrix row indexing.
        valid_chars (set): Set of allowed observed characters.
    """

    def __init__(self, initial_probs, transition_probs, emission_probs):
        """
        Initialize HMM model with probability tables and validate a consistent
        emission alphabet across all states.

        Parameters:
            initial_probs (dict): Starting-state probability table.
            transition_probs (dict:dict): Transition probability table.
            emission_probs (dict:dict): Emission probability table.
        """

        # Store probability tables
        self.initial_probs = initial_probs
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs

        # Convert states to numpy array for row indexing
        self.states = np.array(list(initial_probs.keys()))

        # Infer emission alphabet from first state
        first_state = next(iter(emission_probs))
        alphabet = set(emission_probs[first_state].keys())

        # Validate that all states share the same emission alphabet
        for state, table in emission_probs.items():
            if set(table.keys()) != alphabet:
                raise ValueError(
                    f"Emission alphabet mismatch in state '{state}'. "
                    f"Expected {alphabet}, got {set(table.keys())}"
                )

        # Store validated alphabet for model-level invariants
        self.valid_chars = alphabet

        self.trans_matrix = StrMatrix(transition_probs)
        self.emit_matrix = StrMatrix(emission_probs)


    def initialise_matrix(self, observation, fill_value, dtype):

        # validate the observation sequence
        observation = self.validate_observation(observation)

        # get number of rows in the matrix
        num_states = len(self.states)

        # get number of columns in the matrix
        num_cols = len(observation)

        matrix = np.full((num_states, num_cols), fill_value, dtype=dtype)

        return matrix
    

    def validate_observation(self, observation):
        """
        Validate that all characters in the observation sequence belong to the
        model's emission alphabet.

        Parameters:
            observation (str): Raw input sequence.

        Returns:
            str: Same sequence if valid.

        Raises:
            ValueError: If any character is not in valid_chars.
        """

        for c in observation:
            if c not in self.valid_chars:
                raise ValueError(
                    f"Invalid observed character '{c}' not in valid alphabet {self.valid_chars}"
                )

        return observation
    

    def fwd(self, seq):
        print(seq, "\n")
        FWD_matrix = self.initialise_matrix(seq, 0, np.float64)
        first_char = seq[0]
        print(self.emit_matrix, "\n")
        initial_col = self.emit_matrix[:, first_char]

        key_list = list(self.initial_probs.keys()) # setting the initial probs using the dict
        for i in range(len(self.initial_probs)):   # need to align the keys to the index
            state_i = key_list[i]
            FWD_matrix[i, 0] = np.log(self.initial_probs[state_i]) + initial_col[i] # using np.log because of raw probabilities
        
        with np.printoptions(linewidth=300):
            print(FWD_matrix)
        

    def viterbi_algorithm(self, observation):
        """
        Compute the most probable hidden-state sequence using the Viterbi
        algorithm in natural log (ln) space.

        Parameters:
            observation (str): Observation sequence.

        Returns:
            np.ndarray: Most likely sequence of hidden states.
        """

        # Validate input sequence
        observation = self.validate_observation(observation)

        num_states = len(self.states)
        num_cols = len(observation)

        # Allocate DP matrix (Viterbi) with header row/column
        viterbi_matrix = np.zeros((num_states + 1, num_cols + 1), dtype=object)

        # Fill column headers with observed characters
        for j, obs_char in enumerate(observation):
            viterbi_matrix[0][j + 1] = obs_char

        # Fill row headers with state labels
        for i, state in enumerate(self.states):
            viterbi_matrix[i + 1][0] = str(state)

        viterbi_matrix[0][0] = ""

        # Allocate traceback matrix as empty with same shape
        traceback_matrix = np.empty((num_states + 1, num_cols + 1), dtype=object)

        for j, obs_char in enumerate(observation):
            traceback_matrix[0][j + 1] = obs_char

        for i, state in enumerate(self.states):
            traceback_matrix[i + 1][0] = str(state)

        traceback_matrix[0][0] = ""

        # Initialization step (first column)
        for i, state in enumerate(self.states):
            row = i + 1

            # ln P(state)
            initial_ln = np.log(self.initial_probs[state])

            # ln P(observation[0] | state)
            first_obs = observation[0]
            emission_ln = np.log(self.emission_probs[state][first_obs])

            # ln P(state, observation[0]) = ln(initial) + ln(emission)
            viterbi_matrix[row][1] = initial_ln + emission_ln

            # Traceback pointer is the state itself
            traceback_matrix[row][1] = str(state)

        # DP recurrence
        for j in range(2, num_cols + 1):
            curr_char = observation[j - 1]

            for i, state in enumerate(self.states):
                row = i + 1

                # ln scores from previous column (all states)
                prev_scores = viterbi_matrix[1:, j - 1].astype(float)

                # ln P(state | prev_state) for all previous states
                transition_lns = np.array([
                    np.log(self.transition_probs[prev_state][state])
                    for prev_state in self.states
                ])

                # ln P(curr_char | current state)
                emission_ln = np.log(self.emission_probs[state][curr_char])

                # Candidate ln scores for each previous state
                scores = prev_scores + transition_lns + emission_ln

                # Best previous state
                best_idx = np.argmax(scores)
                best_ln = scores[best_idx]
                best_state = self.states[best_idx]

                # Store best ln probability and traceback pointer
                viterbi_matrix[row][j] = best_ln
                traceback_matrix[row][j] = best_state

        # Termination: choose best final state
        last_column = viterbi_matrix[1:, num_cols].astype(float)
        last_idx = np.argmax(last_column)
        current_state = traceback_matrix[last_idx + 1][num_cols]

        # Allocate path array
        path = np.full(num_cols, "", dtype=object)
        path[-1] = str(current_state)

        state_list = list(self.states)

        # Traceback from last column to first
        for j in range(num_cols - 1, -1, -1):
            curr_label = path[j]
            row = state_list.index(curr_label) + 1
            col = j + 1

            if j > 0:
                path[j - 1] = str(traceback_matrix[row][col])

        # Output matrices and final path
        print(tabulate(viterbi_matrix, tablefmt="plain"))
        print(tabulate(traceback_matrix, tablefmt="plain"))
        print(path)

        return viterbi_matrix, traceback_matrix, path
    


if __name__=="__main__":
    obs = "GGCACTCTCGAA"

    init_probs = {
        "I": 0.5,
        "E": 0.5
    }

    trans_probs = {
        "I": {"I": 0.6, "G": 0.4},
        "E": {"I": 0.4, "G": 0.6}
    }

    emit_probs = {
        "I": {"A": 0.125, "C": 0.5, "G": 0.125, "T": 0.25},
        "E": {"A": 0.5, "C": 0.125, "G": 0.25, "T": 0.125}
    }

    model = HMMModel(init_probs, trans_probs, emit_probs)

    model.fwd(obs)

    # vmat, tmat, path = model.viterbi_algorithm(obs)

    # display(show_matrix_html(vmat))
    # display(show_matrix_html(tmat, float_format="{}"))
    # print(path)

