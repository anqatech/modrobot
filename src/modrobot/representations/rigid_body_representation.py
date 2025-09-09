import numpy as np
from .rotation_matrix import RotationMatrix
from .position_vector import PositionVector
from modrobot.utilities.skew_utils import skew_matrix


class RigidBodyRepresentation:
    __slots__ = (
        "_rotation_object",
        "_position_object",
        "_transformation_matrix",
        "_transformation_matrix_inverse",
        "_adjoint_representation",
        "_adjoint_representation_inverse",
    )
    
    def __init__(self, rotation_matrix, origin_position, check=True):
        self._rotation_object = RotationMatrix(rotation_matrix, check=check)
        self._position_object = PositionVector(origin_position)
        
        # Build transformation matrices
        self._transformation_matrix = self._build_transformation_matrix()
        self._transformation_matrix_inverse = self._build_transformation_matrix_inverse()
        self._adjoint_representation = self._build_adjoint_representation()
        self._adjoint_representation_inverse = self._build_adjoint_representation_inverse()
    
    @classmethod
    def from_space_data(cls, space_rotation_matrix, space_origin_position):
        if space_rotation_matrix.shape != (3, 3):
            raise ValueError("The space rotation matrix must be of dimension 3x3.")
        if space_origin_position.shape != (3, 1):
            raise ValueError("The space origin vector must be of dimension 3x1.")
        
        return cls(space_rotation_matrix, space_origin_position)

    @classmethod
    def from_body_data(cls, body_rotation_matrix, body_origin_position):
        if body_rotation_matrix.shape != (3, 3):
            raise ValueError("The body rotation matrix must be of dimension 3x3.")
        if body_origin_position.shape != (3, 1):
            raise ValueError("The body origin vector must be of dimension 3x1.")
        
        return cls(body_rotation_matrix.T, -body_rotation_matrix.T @ body_origin_position)

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
    
    @property
    def adjoint_representation(self):
        return self._adjoint_representation
    
    @property
    def adjoint_representation_inverse(self):
        return self._adjoint_representation_inverse

    def _build_adjoint_representation(self):
        R = self.rotation_matrix
        p = self.origin_position
        skew_p = skew_matrix(p)
        
        return np.block([
            [     R,        np.zeros((3, 3)) ],
            [ skew_p @ R ,         R         ]
        ])

    def _build_adjoint_representation_inverse(self):
        R = self.rotation_matrix
        p = self.origin_position
        skew_p = skew_matrix(p)
        
        return np.block([
            [     R.T,        np.zeros((3, 3)) ],
            [ -R.T @ skew_p ,        R.T       ]
        ])

    def _build_transformation_matrix(self):
        R = self.rotation_matrix
        p = self.origin_position
        
        return np.block([
            [        R,              p  ],
            [ np.zeros( (1, 3) ) ,   1  ]
        ])

    def _build_transformation_matrix_inverse(self):
        R = self.rotation_matrix
        p = self.origin_position
        
        return np.block([
            [        R.T,             -R.T @ p],
            [ np.zeros( (1, 3) ) ,        1   ]
        ])

    def __repr__(self):
        rotation_matrix_str = np.array2string(
            self.rotation_matrix,
            separator=", ",
            precision=16,
            suppress_small=False,
        )
        origin_position_str = np.array2string(
            self.origin_position,
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
            self.transformation_matrix,
            precision=4,
            suppress_small=True,
            separator=" ",
        )
        return f"Transformation Matrix:\n\n{transformation_matrix_str}"

    def transform_vector(self, vector):
        if vector.shape != (3, 1):
            raise ValueError("The input vector must be of dimension 3x1.")
            
        return self.transformation_matrix @ np.vstack( (vector, 1) )
