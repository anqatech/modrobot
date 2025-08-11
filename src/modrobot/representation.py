import numpy as np

class ModRobot:
    def __init__(self, rotation_matrix, origin_position, check=True):
        if rotation_matrix.shape != (3, 3):
            raise ValueError("The rotation matrix must be of dimension 3x3.")
        if origin_position.shape != (3, 1):
            raise ValueError("The position vector must be of dimension 3x1.")
        
        if check:
            if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=1e-8):
                raise ValueError("The rotation matrix must be orthonormal.")
            if np.linalg.det(rotation_matrix) < 0.0:
                raise ValueError("The rotation matrix must have determinant equal to +1.")
        
        self.rotation_matrix = rotation_matrix
        self.origin_position = origin_position

        self.transformation_matrix = self.build_transformation_matrix()
        self.transformation_matrix_inverse = self.build_transformation_matrix_inverse()

    def build_transformation_matrix(self):
        return np.block([
            [self.rotation_matrix, self.origin_position],
            [ np.zeros( (1, 3) ) ,          1          ]
        ])
    
    def build_transformation_matrix_inverse(self):
        return np.block([
            [self.rotation_matrix.T, -self.rotation_matrix.T @ self.origin_position],
            [ np.zeros( (1, 3) ) ,                          1                      ]
        ])
