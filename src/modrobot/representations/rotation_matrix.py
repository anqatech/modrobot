import numpy as np

class RotationMatrix:
    __slots__ = ("_rotation_matrix",)

    def __init__(self, rotation_matrix, check=True):
        if not isinstance(rotation_matrix, np.ndarray) or rotation_matrix.shape != (3, 3):
            raise TypeError("Input must be a 3x3 NumPy array.")

        if check:
            if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=1e-8):
                raise ValueError("The rotation matrix must be orthonormal (R^T * R = I).")
            if not np.isclose(np.linalg.det(rotation_matrix), 1.0):
                raise ValueError("The determinant of the rotation matrix must be +1.")

        self._rotation_matrix = rotation_matrix

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

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
