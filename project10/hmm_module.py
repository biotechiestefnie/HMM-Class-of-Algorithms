# Import statements
import numpy as np  # for using numpy arrays as matrices
from dict_maker import StrMatrix
from tqdm import tqdm
import random
from scipy.special import logsumexp
from copy import deepcopy
import matplotlib.pyplot as plt

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

    def __init__(self, initial_probs=None, transition_probs=None, emission_probs=None):
        if initial_probs is not None and transition_probs is not None and emission_probs is not None:
            self.initialise_with_prob_dicts(self, initial_probs, transition_probs, emission_probs)

    def initialise_with_prob_dicts(self, initial_probs, transition_probs, emission_probs):
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
        
        #added these for compatibility with 
        state_names = list(emission_probs.keys())
        vocab = list(emission_probs[first_state].keys())
        self.states = np.array(state_names)
        self.vocab_idx = {k: v for v, k in enumerate(vocab)}
        self.state_idx = {k: v for v, k in enumerate(self.states)}

    def random_initialise(self, vocab: list, state_names: list = None, n_states: int=0):
        """
        Initialise using input parameters. Ensure that vocab is a list to preserve order.
        Params:
            n_states: int
            vocab: list(str)
        Returns:
            None
        """

        if state_names is None:
            state_names = random.choices(string.ascii_lowercase, k=len(n_states))
        else:
            n_states = len(state_names)
        
        def recursive_init(remainder, ptr, prob_dict):
            if ptr == len(prob_dict) - 1:
                prob_dict[list(prob_dict.keys())[ptr]] = remainder
                return prob_dict
            prob = np.random.uniform(0, remainder)
            prob_dict[list(prob_dict.keys())[ptr]] = prob
            prob_dict = recursive_init(remainder - prob, ptr + 1, prob_dict)
            return prob_dict
        
        def random_init(keys):
            probs = np.random.dirichlet(np.ones(len(keys)))
            return {k: np.log(p) for k, p in zip(keys, probs)}
        
        pi = {key: 0.0 for key in state_names}
        self.initial_probs = recursive_init(1.0, 0, pi)
        self.initial_probs = {k: np.log(v) for k, v in self.initial_probs.items()} # manually put into log space

        self.emission_probs = {state: {key: 0.0 for key in vocab} for state in state_names}
        for state in self.emission_probs:
            self.emission_probs[state] = recursive_init(1.0, 0, self.emission_probs[state])

        self.transition_probs = {state: {key: 0.0 for key in state_names} for state in state_names}
        for state in self.transition_probs:
            self.transition_probs[state] = recursive_init(1.0, 0, self.transition_probs[state])

        self.valid_chars = set(vocab)
        self.trans_matrix = StrMatrix(self.transition_probs)
        self.emit_matrix = StrMatrix(self.emission_probs)
        
        self.states = np.array(state_names)
        self.vocab_idx = {k: v for v, k in enumerate(vocab)}
        self.state_idx = {k: v for v, k in enumerate(self.states)}
    

    def __str__(self):
        return f"""== TTH, EA, SM ==\n
HIDDEN MARKOV MODEL\n
Initial probabilities: {self.initial_probs}\n\n
Transition probabilities:\n {self.trans_matrix}\n\n
Emission probabilities:\n {self.emit_matrix}\n\n
Available methods: {dir(HMMModel)}

"""


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
        fwd_matrix[:, 0] = np.array([self.initial_probs[s] for s in self.states]) + self.emit_matrix[:, first_char]

        for j in range(1, len(observation)):
            current_char = observation[j]
            emit_col = self.emit_matrix[:, current_char]

            for i, state in enumerate(self.states):
                prev_scores = fwd_matrix[:, j - 1]
                # column slice
                transitions = self.trans_matrix[:, state]
                emission = emit_col[i]
                scores = prev_scores + transitions
                fwd_matrix[i, j] = np.logaddexp.reduce(scores) + emission #particular emission needs to be outside of the sum

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
            current_char = reverse_obs[j - 1]
            emit_col = self.emit_matrix[:, current_char]

            for i, state in enumerate(self.states):
                prev_scores = bwd_matrix[:, j - 1]
                # column slice
                transitions = self.trans_matrix[state, :]
                scores = prev_scores + transitions + emit_col
                bwd_matrix[i, j] = np.logaddexp.reduce(scores)  #can include the whole emission column in the calculation

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

        # compute the average of the last and first columns
        total_prob = self.average_fbw_prob(fwd_matrix, bwd_matrix)

        # vectorized operation to do all calculations
        forward_backward_matrix = (fwd_matrix + bwd_matrix) - total_prob

        # grab the state indices from the matrix using argmax
        state_indices = np.argmax(forward_backward_matrix, axis=0)

        return self.states[state_indices], forward_backward_matrix

    def average_fbw_prob(self, fwd_mtx, bwd_mtx):
        total_prob_fwd = np.logaddexp.reduce(fwd_mtx[:, -1])
        total_prob_bwd = np.logaddexp.reduce(bwd_mtx[:, 0]) # this 
        total_prob = logsumexp([total_prob_fwd, total_prob_bwd]) - np.log(2)
        return total_prob
    

    def get_seq_prob(self, observation):
        fwd_matrix = self.forward_algorithm(observation)    # get the fwd matrix for total prob
        prob = np.logaddexp.reduce(fwd_matrix[:, -1])       # log-add-exp the last column of the matrix
        return prob
    
    def baum_welch_algorithm(self, observations: list, tol=1e-10, pseudocount=-700, max_iter=1e5):

        def check_rows_sum_to_one(matrix, name="matrix", tol=1e-6):
            for i, row in enumerate(matrix):
                log_sum = logsumexp(row)
                if not np.isclose(log_sum, 0.0, atol=tol):  # log(1) = 0
                    raise ValueError(f"{name} row {i}: logsumexp={log_sum}, expected 0.0")

        finit = False
        iter = 0
        average_probs = []

        while not finit and iter < max_iter:

            N = len(self.states)

            pi = np.full(len(self.states), pseudocount, dtype=np.float64)
            trans = StrMatrix({state: {key:pseudocount for key in self.states} for state in self.states}, set_log=False)
            emit = StrMatrix({state: {key:pseudocount for key in self.vocab_idx} for state in self.states}, set_log=False)

            # outer = pseudocount
            prob_avg = []

            for observation in tqdm(observations):
                fwd = self.forward_algorithm(observation)
                bwd = self.backward_algorithm(observation)
                p_seq = logsumexp(fwd[:, -1])
                gamma = fwd + bwd - p_seq #normalizing only by the probability of the sequence in this code
                                          #could have normalized by summing up "outer" like marcus did but this seemed more complicated
                
                prob_avg.append(p_seq)
                
                # outer = logsumexp([outer, p_seq]) # update outer # old code with double normalization just for reference
                
                np.logaddexp(pi, # our accumulated initial probs
                             gamma[:, 0], #first col of our gamma matrix
                             out=pi)

                for j in range(len(observation)):
                    obs = observation[j]
                    emit[:, obs] = np.logaddexp(emit[:, obs], #the accumulated emission probs for that observation
                                                gamma[:, j]) #jth column of the gamma matrix

                    if j < len(observation) - 1:
                        next_obs = observation[j + 1]
                        fwd_t_j = fwd[:, j].reshape(-1, 1) # need to reshape/transpose because these are flat and won't broadcast correctly
                        print(fwd_t_j.shape)
                        em = self.emit_matrix[:, next_obs].reshape(1, -1) # column of the emission matrix corresponding with j + 1. also need to reshape
                        print(em.shape)
                        bwd_j = bwd[:, j + 1] # next col of backwards matrix
                        print(bwd_j.shape)
                        trans_adj = fwd_t_j + self.trans_matrix + em + bwd_j - p_seq
                        input()
                        '''
                        What is trans_adj? (assuming 3 states)

                        [fwd_1j][fwd_1j][fwd_1j]     [trans_00][trans_01][trans_02]     [emit_1j+1][emit_2j+1][emit_3j+1]     [bwd_1j+1][bwd_2j+1][bwd_3j+1]
                        [fwd_2j][fwd_2j][fwd_2j]  x  [trans_10][trans_11][trans_12]  x  [emit_1j+1][emit_2j+1][emit_3j+1]  x  [bwd_1j+1][bwd_2j+1][bwd_3j+1]  /  p_seq (scalar)
                        [fwd_3j][fwd_3j][fwd_3j]     [trans_20][trans_21][trans_22]     [emit_1j+1][emit_2j+1][emit_3j+1]     [bwd_1j+1][bwd_2j+1][bwd_3j+1]

                        flipped         transition matrix               column of emission matrix flipped   backward column (flat by default)
                        fwd jth
                        col
                        '''
                        np.logaddexp(trans, #every emission adjusts the whole transition matrix by some small amount
                                     trans_adj, #amount to adjust the transition matrix
                                     out=trans.matrix) #need to specify output to the matrix directly for the StrMatrix obj
                
                average_probs.append(np.mean(prob_avg)) #update average probs with average for tracking convergence

            #normalize all probabilities since we accumulated by adding
            pi -= logsumexp(pi)
            trans.matrix -= logsumexp(trans.matrix, axis=1, keepdims=True)
            emit.matrix -= logsumexp(emit.matrix, axis=1, keepdims=True)

            #copy and deepcopy to preserve previous iteration
            pi_a = pi.copy()
            trans_a = deepcopy(trans)
            emit_a = deepcopy(emit)

            #re-cast the matrices used for calculations to their updated counterparts
            self.initial_probs = {k: pi_a[r] for r, k in enumerate(self.states)}
            self.trans_matrix = trans_a
            self.emit_matrix = emit_a

            # checking if the probs add to 1 (debugging)
            init_sum = logsumexp(list(self.initial_probs.values()))
            if not np.isclose(init_sum, 0.0, atol=1e-6):
                raise ValueError(f"initial probs: logsumexp={init_sum}, expected 0.0 (sum={np.exp(init_sum)})")
            check_rows_sum_to_one(self.trans_matrix.matrix)
            check_rows_sum_to_one(self.emit_matrix.matrix)

            if iter > 0: #check for convergence
                if np.allclose(pi, pi_a, atol=tol) \
                    and np.allclose(trans, trans_a, atol=tol) \
                    and np.allclose(emit, emit_a, atol=tol):
                    finit=True

            iter += 1

        plt.plot(average_probs)
        plt.xlabel("Iteration")
        plt.ylabel("Log-likelihood")
        plt.title("Baum-Welch Convergence")
        plt.show()




if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet("genome_data.parquet")
    data = df["seq"].to_list()[:20]

    np.random.seed=42
    HMM = HMMModel()
    HMM.random_initialise(["A", "T", "C", "G", "N"], ["bingo", "bango", "bongo", "leave", "the", "congo"])
    HMM.baum_welch_algorithm(data, max_iter=1000, tol=1e-20)
    print(HMM)

