import numpy as np

class RigidBodyRepresentation:
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
    
    @classmethod
    def from_transformation_matrix(cls, transformation_matrix, check=True):
        if transformation_matrix.shape != (4, 4):
            raise ValueError("The transformation matrix must be of dimension 4x4.")
        if check and not np.allclose(transformation_matrix[3], [0, 0, 0, 1], atol=1e-12):
            raise ValueError("Bottom row of the transformation matrix must be [0, 0, 0, 1].")
        
        R = transformation_matrix[:3, :3]
        p = transformation_matrix[:3, 3:4]
        return cls(R, p, check=check)

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

    def __repr__(self):
        rotation_matrix_str = np.array2string(
            self._rotation_matrix,
            separator=", ",
            precision=16,
            suppress_small=False,
        )
        origin_position_str = np.array2string(
            self._origin_position,
            separator=", ",
            precision=16,
            suppress_small=False,
        )
        return (
            f"ModRobot(rotation_matrix=np.array({rotation_matrix_str}), "
            f"origin_position=np.array({origin_position_str}))"
        )
        
    def __str__(self):
        rotation_matrix_str = np.array2string(
            self._rotation_matrix,
            precision=4,
            suppress_small=True,
            separator=" ",
        )
        origin_position_str = np.array2string(
            self._origin_position,
            precision=4,
            suppress_small=True,
            separator=" ",
        )
        transformation_matrix_str = np.array2string(
            self._transformation_matrix,
            precision=4,
            suppress_small=True,
            separator=" ",
        )
        detR = float(np.linalg.det(self._rotation_matrix))
        return (
            "ModRobot(SE3):\n\n"
            f"R:\n{rotation_matrix_str}\n\n"
            f"p:\n{origin_position_str}\n\n"
            "T:\n"
            f"{transformation_matrix_str}"
        )

    def transform_vector(self, vector):
        if vector.shape != (3, 1):
            raise ValueError("The input vector must be of dimension 3x1.")
            
        return self.transformation_matrix @ np.vstack( (vector, 1) )
