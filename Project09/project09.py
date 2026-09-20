# Import statements
import numpy as np  # for using numpy arrays as matrices
from dict_maker import StrMatrix


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

        # enforce consistent, predictable state order
        self.states = np.array(sorted(initial_probs.keys()))

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
        """
        Initialize a DP matrix for a given observation sequence to be used by each algorithm
        Parameters:
            observation (str): Observation sequence
            fill_value (float): Initial fill value for all cells
            dtype (type): Numpy dtype for the matrix
        Returns:
            np.ndarray: Matrix of shape (num_states, len(observation))
        """

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
            observation (str): Raw input sequence
        Returns:
            observation (str): Same sequence if valid
        Raises:
            ValueError: If any character is not in valid_chars
        """
        # Compare observation sequence to valid characters
        for c in observation:
            if c not in self.valid_chars:
                # If nonvalid character identified, raise error
                raise ValueError(
                    f"Invalid observed character '{c}' not in valid alphabet {self.valid_chars}"
                )
        # Return observation string if no invalid characters identified
        return observation


    def viterbi_algorithm(self, observation):
        """
        Compute the overall optimal hidden-state sequence path using the Viterbi algorithm
        in log-space for numerical stability. Uses dynamic programming for traceback matrix
        to identify optimal sequence.
        Parameters:
            observation (str): Observation sequence consisting of characters from the emission alphabet
        Returns:
            vmat (np.ndarray): DP matrix of shape (num_states, T) containing the best log-probabilities
                                         for each state at each position
            tmat (np.ndarray): Matrix of shape (num_states, T) storing the previous state label that
                                           produced the maximum score at each position
            path (list[str]): The most likely hidden-state sequence (Viterbi path)
        """

        # Validate observation sequence
        observation = self.validate_observation(observation)

        # Number of states and sequence length
        num_states = len(self.states)
        T = len(observation)

        # Allocate DP matrix for log-probabilities
        # Fill with -inf to represent impossible paths
        vmat = self.initialise_matrix(observation, fill_value=-np.inf, dtype=np.float64)

        # Allocate traceback matrix (stores state labels)
        tmat = np.full((num_states, T), None, dtype=object)

        # Initialization
        first_char = observation[0]
        emit_col = self.emit_matrix[:, first_char]  # log P(obs | state)

        for i, state in enumerate(self.states):
            # log P(state) + log P(first observation | state)
            vmat[i, 0] = np.log(self.initial_probs[state]) + emit_col[i]
            tmat[i, 0] = str(state)

        # Recursion
        for t in range(1, T):
            curr_char = observation[t]
            emit_col = self.emit_matrix[:, curr_char]  # log P(obs_t | state)

            for i, curr_state in enumerate(self.states):
                # Previous column scores
                prev_scores = vmat[:, t - 1]

                # Transition log-probabilities into curr_state
                trans_col = self.trans_matrix[:, curr_state]

                # Candidate scores for all previous states
                scores = prev_scores + trans_col

                # Best previous state
                best_prev_index = np.argmax(scores)
                vmat[i, t] = scores[best_prev_index] + emit_col[i]
                tmat[i, t] = str(self.states[best_prev_index])

        # Best final state termination and traceback
        last_state_index = np.argmax(vmat[:, -1])
        path = [str(self.states[last_state_index])]

        # Trace backward from t = T-1 down to t = 1
        for t in range(T - 1, 0, -1):
            prev_state = tmat[last_state_index, t]
            path.append(prev_state)
            last_state_index = np.where(self.states == prev_state)[0][0]

        # Reverse to get left-to-right order
        path.reverse()

        # Force traceback matrix to plain Python strings for clean display
        tmat = np.vectorize(str)(tmat)

        return vmat, tmat, path




    def forward_algorithm(self, observation):
        """
        Compute the forward (alpha) matrix in log-space.
        Parameters:
            observation (str): Observation sequence.
        Returns:
            fwd_matrix (np.ndarray): Forward probability matrix.
        """

        # Initialise the forward matrix with 0s
        fwd_matrix = self.initialise_matrix(observation, 0, np.float64)

        # Getting the first character from the observation sequence
        first_char = observation[0]

        # Getting the emission prob for the first character
        initial_col = self.emit_matrix[:, first_char]

        key_list = list(self.initial_probs.keys())

        for i in range(len(self.initial_probs)):
            # Aligning the initial prob keys to the index
            state_i = key_list[i]
            # First column calculation
            # using np.log because of raw probabilities and to prevent underflow
            fwd_matrix[i, 0] = np.log(self.initial_probs[state_i]) + initial_col[i]

        for j in range(1, len(observation)):
            current_char = observation[j]
            emit_col = self.emit_matrix[:, current_char]

            for i, state in enumerate(self.states):
                prev_scores = fwd_matrix[:, j - 1]
                # column slice
                transitions = self.trans_matrix[:, state]
                emission = emit_col[i]
                scores = prev_scores + transitions + emission
                fwd_matrix[i, j] = np.logaddexp.reduce(scores)

        return fwd_matrix


    def backward_algorithm(self, observation):
        """
        Compute the backward (beta) matrix in log-space.
        Parameters:
            observation (str): Observation sequence.
        Returns:
            bwd_matrix (np.ndarray): Backward probability matrix.
        """

        # Reverse the observation sequence
        reverse_obs = observation[::-1]

        # Initialize the backward matrix
        bwd_matrix = self.initialise_matrix(reverse_obs, 0, np.float64)

        # First column of reverse matrix = 1
        bwd_matrix[:, 0] = 0  # log(1) = 0

        for j in range(1, len(reverse_obs)):
            current_char = reverse_obs[j]
            emit_col = self.emit_matrix[:, current_char]

            for i, state in enumerate(self.states):
                prev_scores = bwd_matrix[:, j - 1]
                # column slice
                transitions = self.trans_matrix[:, state]
                emission = emit_col[i]
                scores = prev_scores + transitions + emission
                bwd_matrix[i, j] = np.logaddexp.reduce(scores)

        # Reverse back the matrix
        bwd_matrix = bwd_matrix[:, ::-1]

        return bwd_matrix


    def forward_backward_algorithm(self, observation):
        """
        Compute posterior state probabilities using the Forward-Backward algorithm.
        Parameters:
            observation (str): Observation sequence.
        Returns:
            np.ndarray: Most likely state at each position.
            forward_backward_matrix (np.ndarray): Posterior probability matrix
                """

        fwd_matrix = self.forward_algorithm(observation)
        bwd_matrix = self.backward_algorithm(observation)

        forward_backward_matrix = self.initialise_matrix(observation, 0, np.float64)

        total_prob_fwd = np.logaddexp.reduce(fwd_matrix[:, -1])
        total_prob_bwd = np.logaddexp.reduce(bwd_matrix[:, -1])
        total_prob = total_prob_fwd + total_prob_bwd - 2

        for j in range(0, len(observation)):
            for i in range(0, len(self.states)):
                forward_backward_matrix[i, j] = fwd_matrix[i, j] + bwd_matrix[i, j] - total_prob

        state_indices = np.argmax(forward_backward_matrix, axis=0)

        return self.states[state_indices], forward_backward_matrix
