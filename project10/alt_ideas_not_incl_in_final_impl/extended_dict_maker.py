import numpy as np


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

    def __init__(self, prob_dict, set_log=True):
        """
        Initialize the StrMatrix from a nested dictionary.
        Parameters:
            prob_dict (dict:dict): Nested probability dictionary.
            set_log (bool): If True, convert probabilities to log-space.
        """

        outer_keys = list(prob_dict.keys())
        inner_keys = list(next(iter(prob_dict.values())).keys())

        inner_key_set = set(inner_keys)
        # Validate all rows share the same keys
        for key, row_dict in prob_dict.items():
            if set(row_dict.keys()) != inner_key_set:
                raise ValueError(
                    f"Row '{key}' has inconsistent keys. "
                    f"Expected {inner_key_set}, got {set(row_dict.keys())}"
                )

        # ordered list of row labels extracted from outer/inner dictionary keys
        self.row_labels = outer_keys
        self.col_labels = inner_keys

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

        # row is ":" → return full column
        if isinstance(row, slice):
            return self.get_col(col)

        # col is ":" → return full row
        if isinstance(col, slice):
            return self.get_row(row)

        # both are strings → return single value
        i = self.outer_key_map[row]
        j = self.inner_key_map[col]

        return self.matrix[i, j]

