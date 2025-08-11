import numpy as np

class ModRobot:
    __slots__ = (
        "_rotation_matrix",
        "_origin_position",
        "_transformation_matrix",
        "_transformation_matrix_inverse",
    )
    
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
        
        self._rotation_matrix = rotation_matrix
        self._origin_position = origin_position

        self._transformation_matrix = self.build_transformation_matrix()
        self._transformation_matrix_inverse = self.build_transformation_matrix_inverse()

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

    @property
    def origin_position(self):
        return self._origin_position

    @property
    def transformation_matrix(self):
        return self._transformation_matrix
    
    @property
    def transformation_matrix_inverse(self):
        return self._transformation_matrix_inverse
    
    def build_transformation_matrix(self):
        return np.block([
            [self._rotation_matrix, self._origin_position],
            [ np.zeros( (1, 3) ) ,          1          ]
        ])

    def build_transformation_matrix_inverse(self):
        return np.block([
            [self._rotation_matrix.T, -self._rotation_matrix.T @ self._origin_position],
            [ np.zeros( (1, 3) ) ,                          1                      ]
        ])
