
import numpy as np
from tabulate import tabulate




transitions = {"I": {"I": 0.1, "G": 0.9}, 
               "G": {"I": 0.6, "G": 0.4}}
emissions = {"I": {"A": 0.1, "T": 0.1, "C": 0.4, "G": 0.4}, 
             "G": {"A": 0.15, "T": 0.15, "C": 0.3, "G": 0.3}}


class StrMatrix:

    def __init__(self, prob_dict, set_log=True):
        outer_keys = list(prob_dict.keys())
        first_key = outer_keys[0]
        inner_keys = list(prob_dict[first_key].keys())

        self.outer_key_map = {v: i for i, v in enumerate(outer_keys)} # the most critical part of mapping
        self.inner_key_map = {v: i for i, v in enumerate(inner_keys)}

        self.matrix = np.zeros((len(self.outer_key_map), len(self.inner_key_map)))

        for key in outer_keys:
            row = self.outer_key_map[key]
            for other_key in inner_keys:
                col = self.inner_key_map[other_key]
                self.matrix[row, col] = np.log(prob_dict[key][other_key]) if set_log else prob_dict[key][other_key] # end of most critical part

        self.inv_outer_key_map = {i: v for v, i in enumerate(self.outer_key_map.items())}
        self.inv_inner_key_map = {i: v for v, i in enumerate(self.inner_key_map.items())}
    
    def get_col(self, inx: str):
        idx = self.inner_key_map[inx]
        return self.matrix[:, idx]

    def get_row(self, inx: str):
        idx = self.outer_key_map[inx]
        return self.matrix[idx, :]
    
    def get_matrix(self):
        return self.matrix
    
    def __getitem__(self, coords):
        row, col = coords
        if not isinstance(row, slice):
            i = self.outer_key_map[row]
        else:
            return self.get_col(col)
        if not isinstance(col, slice):
            j = self.inner_key_map[col]
        else:
            return self.get_row(row)
        return self.matrix[i, j]
    
    def __str__(self):
        row_labels = list(self.outer_key_map.keys())
        col_labels = list(self.inner_key_map.keys())

        # Build list-of-lists with row label prepended
        labeled = [
            [row_labels[i]] + [f"{self.matrix[i, j]:.3f}" for j in range(len(col_labels))]
            for i in range(len(row_labels))
        ]

        return tabulate(labeled, headers=[""] + col_labels)


if __name__ == "__main__":

    TransitionMatrix = StrMatrix(transitions)

    print(TransitionMatrix)

    print(TransitionMatrix[:, "I"]) # get column "I" only

    print(TransitionMatrix["I", :]) # get row "I" only

    EmissionMatrix = StrMatrix(emissions)

    print(EmissionMatrix)