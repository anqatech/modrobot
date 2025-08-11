import numpy as np

class ModRobot:
    def __init__(self, rotation_matrix, origin_position):
        self.rotation_matrix = rotation_matrix
        self.origin_position = origin_position

        self.transformation_matrix = self.build_transformation_matrix()

    def build_transformation_matrix(self):
        return np.block([
            [self.rotation_matrix, self.origin_position],
            [ np.zeros( (1, 3) ) ,          1          ]
        ])
