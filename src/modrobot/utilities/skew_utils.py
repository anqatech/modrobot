import numpy as np

def skew_matrix(vector):
    v1, v2, v3 = vector.squeeze()

    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0],
    ])

def vector_from_skew_matrix(skew_matrix):
    v1 = skew_matrix[2, 1]
    v2 = skew_matrix[0, 2]
    v3 = skew_matrix[1, 0]
    
    return np.array([
        [v1],
        [v2],
        [v3],
    ])
