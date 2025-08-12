import numpy as np
from .rotation_matrix import RotationMatrix
from .position_vector import PositionVector

class RigidBodyRepresentation:
    __slots__ = (
        "_rotation_object",
        "_position_object",
        "_transformation_matrix",
        "_transformation_matrix_inverse",
    )
    
    def __init__(self, rotation_matrix, origin_position, check=True):
        self._rotation_object = RotationMatrix(rotation_matrix, check=check)
        self._position_object = PositionVector(origin_position)
        
        # Build transformation matrices
        self._transformation_matrix = self._build_transformation_matrix()
        self._transformation_matrix_inverse = self._build_transformation_matrix_inverse()
    
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
        return self._rotation_object.rotation_matrix

    @property
    def origin_position(self):
        return self._position_object.position_vector

    @property
    def transformation_matrix(self):
        return self._transformation_matrix
    
    @property
    def transformation_matrix_inverse(self):
        return self._transformation_matrix_inverse
    
    def _build_transformation_matrix(self):
        R = self._rotation_object.rotation_matrix
        p = self._position_object.position_vector
        
        return np.block([
            [        R,              p  ],
            [ np.zeros( (1, 3) ) ,   1  ]
        ])

    def _build_transformation_matrix_inverse(self):
        R = self._rotation_object.rotation_matrix
        p = self._position_object.position_vector
        
        return np.block([
            [        R.T,             -R.T @ p],
            [ np.zeros( (1, 3) ) ,        1   ]
        ])

    def __repr__(self):
        rotation_matrix_str = np.array2string(
            self._rotation_object.rotation_matrix,
            separator=", ",
            precision=16,
            suppress_small=False,
        )
        origin_position_str = np.array2string(
            self._position_object.position_vector,
            separator=", ",
            precision=16,
            suppress_small=False,
        )
        return (
            f"RigidBodyRepresentation(rotation_matrix=np.array({rotation_matrix_str}), "
            f"origin_position=np.array({origin_position_str}))"
        )
        
    def __str__(self):
        transformation_matrix_str = np.array2string(
            self._transformation_matrix,
            precision=4,
            suppress_small=True,
            separator=" ",
        )
        return f"Transformation Matrix:\n\n{transformation_matrix_str}"

    def transform_vector(self, vector):
        if vector.shape != (3, 1):
            raise ValueError("The input vector must be of dimension 3x1.")
            
        return self.transformation_matrix @ np.vstack( (vector, 1) )
