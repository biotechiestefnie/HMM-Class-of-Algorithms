import numpy as np
from tabulate import tabulate

class StrMatrix:
    """
    Convert a nested probability dictionary into a matrix with
    string-based row and column indexing. Optionally converts
    probabilities to natural log space.
    Attributes:
        outer_key_map (dict): Maps outer keys (rows) to integer indices.
        inner_key_map (dict): Maps inner keys (columns) to integer indices.
        matrix (np.ndarray): Numerical matrix representation of the dictionary.
        inv_outer_key_map (dict): Reverse lookup for row index → key.
        inv_inner_key_map (dict): Reverse lookup for column index → key.
    """

    def __init__(self, prob_dict=None, set_log=True, matrix: np.array=None, outer_key_map=None, inner_key_map=None):
        """
        Initialize the StrMatrix from a nested dictionary.
        Parameters:
            prob_dict (dict:dict): Nested probability dictionary.
            set_log (bool): If True, convert probabilities to log-space.
        """

        if matrix is None:
            # Extract row and column labels
            outer_keys = list(prob_dict.keys())
            first_key = outer_keys[0]
            inner_keys = list(prob_dict[first_key].keys())

            # Map string keys to matrix indices
            self.outer_key_map = {v: i for i, v in enumerate(outer_keys)}  # row mapping
            self.inner_key_map = {v: i for i, v in enumerate(inner_keys)}  # column mapping

            # Allocate matrix
            self.matrix = np.zeros((len(self.outer_key_map), len(self.inner_key_map)))

            # Fill matrix with probabilities (log or raw)
            for key in outer_keys:
                row = self.outer_key_map[key]
                for other_key in inner_keys:
                    col = self.inner_key_map[other_key]
                    value = prob_dict[key][other_key]
                    self.matrix[row, col] = np.log(value) if set_log else value

            # Build inverse maps for lookup
            self.inv_outer_key_map = {i: k for k, i in self.outer_key_map.items()}
            self.inv_inner_key_map = {i: k for k, i in self.inner_key_map.items()}
        else:
            self.matrix = matrix
            self.outer_key_map = outer_key_map
            self.inner_key_map = inner_key_map


    def get_col(self, inx: str):
        """
        Return an entire column by its string label.
        Parameters:
            inx (str): Column key.
        Returns:
            np.ndarray: Column vector.
        """
        idx = self.inner_key_map[inx]
        return self.matrix[:, idx]


    def get_row(self, inx: str):
        """
        Return an entire row by its string label.
        Parameters:
            inx (str): Row key.
        Returns:
            np.ndarray: Row vector.
        """
        idx = self.outer_key_map[inx]
        return self.matrix[idx, :]


    def get_matrix(self):
        """
        Return the underlying numerical matrix.
        Returns:
            np.ndarray: The stored matrix.
        """
        return self.matrix
    
    def __add__(self, other):
        return self.matrix + other
    
    def __radd__(self, other):
        return other + self.matrix
    
    def __sub__(self, other):
        raw = other.matrix if isinstance(other, StrMatrix) else other
        return StrMatrix(matrix=self.matrix - raw, outer_key_map=self.outer_key_map, inner_key_map=self.inner_key_map)

    def __rsub__(self, other):
        raw = other.matrix if isinstance(other, StrMatrix) else other
        return StrMatrix(matrix=raw - self.matrix, outer_key_map=self.outer_key_map, inner_key_map=self.inner_key_map)
    
    def __array__(self):
        return self.matrix
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raw = [x.matrix if isinstance(x, StrMatrix) else x for x in inputs]
        result = getattr(ufunc, method)(*raw, **kwargs)
        return StrMatrix(matrix=result, outer_key_map=self.outer_key_map, inner_key_map=self.inner_key_map)
    
    @property
    def shape(self):
        return self.matrix.shape

    def __getitem__(self, coords):
        """
        Enable matrix-style indexing using string keys.
        Supports:
            matrix["I", "G"] → single value
            matrix["I", :]   → full row
            matrix[:, "G"]   → full column
        Parameters:
            coords (tuple): (row_key or slice, col_key or slice)
        Returns:
            float or np.ndarray: Retrieved value or slice.
        """

        row, col = coords

        # Row is a string → convert to index
        if not isinstance(row, slice):
            i = self.outer_key_map[row]
        else:
            return self.get_col(col)

        # Column is a string → convert to index
        if not isinstance(col, slice):
            j = self.inner_key_map[col]
        else:
            return self.get_row(row)

        return self.matrix[i, j]
    
    def __setitem__(self, coords, value):
        row, col = coords
        if not isinstance(row, slice):
            i = self.outer_key_map[row]
        else:
            self.matrix[:, self.inner_key_map[col]] = value

        # Column is a string → convert to index
        if not isinstance(col, slice):
            j = self.inner_key_map[col]
        else:
            self.matrix[self.outer_key_map[row], :] = value

        # self.matrix[i, j] = value
    

    def __str__(self):
        row_labels = list(self.outer_key_map.keys())
        col_labels = list(self.inner_key_map.keys())

        # Build list-of-lists with row label prepended
        labeled = [
            [row_labels[i]] + [f"{self.matrix[i, j]:.3f}" for j in range(len(col_labels))]
            for i in range(len(row_labels))
        ]

        return tabulate(labeled, headers=[""] + col_labels)
