import numpy as np

class ModRobot:
    def __init__(self, rotation_matrix, origin_position):
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
