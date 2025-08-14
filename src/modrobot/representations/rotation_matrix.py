import numpy as np

class RotationMatrix:
    __slots__ = (
        "_rotation_matrix",
        "_theta",
        "_omega",
    )

    def __init__(self, rotation_matrix, check=True):
        if not isinstance(rotation_matrix, np.ndarray) or rotation_matrix.shape != (3, 3):
            raise TypeError("Input must be a 3x3 NumPy array.")

        if check:
            if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=1e-8):
                raise ValueError("The rotation matrix must be orthonormal (R^T * R = I).")
            if not np.isclose(np.linalg.det(rotation_matrix), 1.0):
                raise ValueError("The determinant of the rotation matrix must be +1.")

        self._rotation_matrix = rotation_matrix
        self._theta, self._omega = self._compute_exponential_coordinates()

    @classmethod
    def from_exponential_coordinates(cls, exponential_coordinates, check=True):
        if exponential_coordinates.shape != (3, 1):
            raise ValueError("The rotation axis exponential coordinates must be of dimension 3x1.")

        theta = np.linalg.norm(exponential_coordinates)
        if theta < 1e-12:
            R = np.eye(3)
            return cls(R, check=check)
        omega = exponential_coordinates / theta
        R = cls.matrix_exponential(theta, omega)
        return cls(R, check=check)

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

    @property
    def theta(self):
        return self._theta

    @property
    def omega(self):
        return self._omega

    def _compute_exponential_coordinates(self):
        if np.allclose(self.rotation_matrix, np.eye(3), atol=1e-8):
            theta = 0.0
            omega = np.array([[0.0], [0.0], [0.0]])
        
        elif np.isclose(np.trace(self.rotation_matrix), -1.0, atol=1e-8):
            theta = np.pi
            R11, R22, R33 = self.rotation_matrix.diagonal()
            
            if R11 >= R22 and R11 >= R33:
                omega = ( 1.0 / np.sqrt(2.0 * (1 + R11)) ) * np.array([
                    R11 + 1.0, 
                    self.rotation_matrix[0,1], 
                    self.rotation_matrix[0,2]
                ])
            elif R22 >= R33:
                omega = ( 1.0 / np.sqrt(2.0 * (1 + R22)) ) * np.array([
                    self.rotation_matrix[0,1], 
                    R22 + 1.0, 
                    self.rotation_matrix[1,2]
                ])
            else:
                omega = ( 1.0 / np.sqrt(2.0 * (1 + R33)) ) * np.array([
                    self.rotation_matrix[0,2], 
                    self.rotation_matrix[1,2], 
                    R33 + 1.0
                ])
                
        else:
            theta = np.arccos(0.5 * (np.trace(self.rotation_matrix) - 1.0))
            skew_omega = ( 1.0 / (2 * np.sin(theta)) ) * (self.rotation_matrix - self.rotation_matrix.T)
            omega = self.vector_from_skew_matrix(skew_omega)

        return theta, omega

    def __repr__(self):
        rotation_matrix_str = np.array2string(
            self.rotation_matrix,
            separator=", ",
            precision=16,
            suppress_small=False,
        )

        return f"RotationMatrix(np.array({rotation_matrix_str}))"

    def __str__(self):
        rotation_matrix_str = np.array2string(
            self.rotation_matrix,
            precision=4,
            suppress_small=True,
            separator=" ",
        )

        return f"Rotation Matrix:\n\n{rotation_matrix_str}"
    
    @staticmethod
    def skew_matrix(vector):
        v1, v2, v3 = vector.squeeze()
    
        return np.array([
            [0, -v3, v2],
            [v3, 0, -v1],
            [-v2, v1, 0],
        ])
    
    @staticmethod
    def vector_from_skew_matrix(skew_matrix):
        v1 = skew_matrix[2, 1]
        v2 = skew_matrix[0, 2]
        v3 = skew_matrix[1, 0]
        
        return np.array([
            [v1],
            [v2],
            [v3],
        ])

    @classmethod
    def matrix_exponential(cls, theta, omega):
        w_skew = cls.skew_matrix(omega)
        return np.eye(3) + np.sin(theta) * w_skew + (1.0 - np.cos(theta)) * w_skew @ w_skew
