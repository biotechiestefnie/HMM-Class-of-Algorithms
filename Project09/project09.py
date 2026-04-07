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
    

    def forward_algorithm (self, observation):

        #print(seq, "\n")
        # Initializing the forward matrix with 0s
        fwd_matrix = self.initialise_matrix(observation, 0, np.float64)

        # Getting the first character from the observation sequence
        first_char = observation[0]

        #print(self.emit_matrix, "\n")

        # Getting the emission prob for the first character
        initial_col = self.emit_matrix[:, first_char]

        key_list = list(self.initial_probs.keys())
        #print(self.trans_matrix.inner_key_map)

        for i in range(len(self.initial_probs)):

            # Aligning the initial prob keys to the index
            state_i = key_list[i]

            # First column calculation
            fwd_matrix[i, 0] = np.log(self.initial_probs[state_i]) + initial_col[i] # using np.log because of raw probabilities

        #print(f"states        : {self.states}")
        #print(f"inner_key_map : {self.trans_matrix.inner_key_map}")
        #print(f"key_list      : {key_list}")

        for j in range(1, len(observation)):

            current_char = observation[j]

            emit_col = self.emit_matrix[:, current_char]

            #print(f"\n--- position j={j}, observed char='{current_char}' ---")

            for i, state in enumerate(self.states):

                prev_scores = fwd_matrix[:, j - 1]
                #print(f"prev='{prev_scores}' ---")

                # column slice
                transitions = self.trans_matrix[:, state]

                emission = emit_col[i]
                #print(f"emission={emission}")

                scores = prev_scores + transitions + emission
                #print(scores)

                fwd_matrix[i, j] = np.logaddexp.reduce(scores)

        print(tabulate(fwd_matrix, tablefmt="pretty"))
        return fwd_matrix

    def backward_algorithm (self, observation):

        # Reverse the observation sequence
        reverse_obs = observation[::-1]

        # Initialize the backward matrix
        bwd_matrix = self.initialise_matrix(reverse_obs, 0, np.float64)

        # First column of reverse matrix = 1
        bwd_matrix[:, 0] = 0 # log(1) = 0

        for j in range(1, len(reverse_obs)):

            current_char = reverse_obs[j]

            emit_col = self.emit_matrix[:, current_char]

            #print(f"\n--- position j={j}, observed char='{current_char}' ---")

            for i, state in enumerate(self.states):

                prev_scores = bwd_matrix[:, j - 1]
                #print(f"prev='{prev_scores}' ---")

                # column slice
                transitions = self.trans_matrix[:, state]

                emission = emit_col[i]
                #print(f"emission={emission}")

                scores = prev_scores + transitions + emission
                #print(scores)

                bwd_matrix[i, j] = np.logaddexp.reduce(scores)

        # Reverse back the matrix
        bwd_matrix = bwd_matrix[:, ::-1]

        print(tabulate(bwd_matrix, tablefmt="pretty"))
        return bwd_matrix

    def forward_backward_algorithm(self, observation):

        fwd_matrix = self.forward_algorithm(observation)
        bwd_matrix = self.backward_algorithm(observation)

        forward_backward_matrix = self.initialise_matrix(observation, 0, np.float64)

        total_prob_fwd = np.logaddexp.reduce(fwd_matrix[:, -1])
        total_prob_bwd = np.logaddexp.reduce(bwd_matrix[:, -1])
        total_prob = total_prob_fwd + total_prob_bwd - 2

        for j in range(0, len(observation)):
            for i in range(0, len(self.states)):
                forward_backward_matrix[i, j] = fwd_matrix[i, j] + bwd_matrix[i, j] - total_prob

        print(tabulate(forward_backward_matrix, tablefmt="pretty"))
        return forward_backward_matrix

if __name__=="__main__":
    obs = "ATGCAA"

    init_probs = {
        "E": 0.6,
        "I": 0.4
    }

    trans_probs = {
        "E": {"E": 0.8, "I": 0.2},
        "I": {"E": 0.3, "I": 0.7}
    }

    emit_probs = {
        "E": {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3},
        "I": {"A": 0.1, "C": 0.4, "G": 0.4, "T": 0.1}
    }

    model = HMMModel(init_probs, trans_probs, emit_probs)

    model.forward_backward_algorithm(obs)


    # vmat, tmat, path = model.viterbi_algorithm(obs)

    # display(show_matrix_html(vmat))
    # display(show_matrix_html(tmat, float_format="{}"))
    # print(path)

#Transition matrix:
    #E        I
#--  ------  ------
#E - 0.223 - 1.609
#I - 1.204 - 0.357

#Emission matrix:
        #A      C       G       T
#--  ------  ------  ------  ------
#E - 1.204 - 1.609 - 1.609 - 1.204
#I - 2.303 - 0.916 - 0.916 - 2.303