import numpy as np
from .rigid_body_representation import RigidBodyRepresentation
from .rotation_matrix import RotationMatrix


class RigidBodyTwist:
    __slots__ = (
        "_body_twist",
        "_representation",
        "_space_twist",
    )

    def __init__(self, twist_vector, representation, twist_type):
        if not isinstance(twist_vector, np.ndarray) or twist_vector.shape != (6, 1):
            raise TypeError("The body twist must be a 6x1 NumPy array.")
        if not isinstance(representation, RigidBodyRepresentation):
            raise TypeError("The rigid body representation must be of type RigidBodyRepresentation.")
    
        self._representation = representation
        twist_type = twist_type.lower()

        if twist_type == "body":
            self._body_twist = twist_vector
            self._space_twist = self.representation.adjoint_representation @ self.body_twist
        elif twist_type == "space":
            self._space_twist = twist_vector
            self._body_twist = self.representation.adjoint_representation_inverse @ self.space_twist
        else:
            raise ValueError("twist_type must be either 'body' or 'space'.")

    @classmethod
    def from_exponential_coordinates(cls, exponential_coordinates, twist_type):
        if exponential_coordinates.shape != (6, 1):
            raise ValueError("The rotation axis exponential coordinates must be of dimension 6x1.")
        if not twist_type.lower() == "space":
            raise ValueError(
                f"When creating a RigidBodyTwist from exponential coordinates "
                f"the twist_type must be 'space'."
            )
        
        w = exponential_coordinates[0:3]
        R = RotationMatrix.from_exponential_coordinates(w)
        v = exponential_coordinates[3:]
        if not np.isclose(R.theta, 0.0, atol=1e-12):
            p = cls.top_right_matrix_exponential(w) @ (v / R.theta)
        else:
            p = v
        matrix_exponential = np.block([
            [R.rotation_matrix, p],
            [np.array([[0, 0, 0, 1]])],
        ])
        T = RigidBodyRepresentation.from_transformation_matrix(matrix_exponential)

        return cls(exponential_coordinates, T, twist_type)

    @property
    def body_twist(self):
        return self._body_twist

    @property
    def body_twist_matrix(self):
        return self.body_twist_skew_matrix()
    
    @property
    def space_twist(self):
        return self._space_twist
    
    @property
    def space_twist_matrix(self):
        return self.space_twist_skew_matrix()
    
    @property
    def representation(self):
        return self._representation
    
    def __repr__(self):
        body_twist_str = np.array2string(
            self.body_twist,
            separator=", ",
            precision=16,
            suppress_small=False,
        )

        return f"RigidBodyTwist(np.array({body_twist_str}), twist_type='body')"

    def __str__(self):
        body_twist_str = np.array2string(
            self.body_twist,
            precision=4,
            suppress_small=True,
            separator=" ",
        )
        space_twist_str = np.array2string(
            self.space_twist,
            precision=4,
            suppress_small=True,
            separator=" ",
        )

        return f"Body Twist:\n\n{body_twist_str}\n\nSpace Twist:\n\n{space_twist_str}"

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
    def top_right_matrix_exponential(cls, w):
        theta = np.linalg.norm(w)
        omega = w / theta
        w_skew = cls.skew_matrix(omega)
        return np.eye(3) * theta + (1 - np.cos(theta)) * w_skew + (theta - np.sin(theta)) * w_skew @ w_skew
    
    def body_twist_skew_matrix(self):
        w = self.body_twist[0:3]
        v = self.body_twist[3:]
    
        return np.block([ 
            [self.skew_matrix(w), v] , 
            [np.zeros( (1, 4) )]
        ])

    def space_twist_skew_matrix(self):
        w = self.space_twist[0:3]
        v = self.space_twist[3:]
    
        return np.block([ 
            [self.skew_matrix(w), v] , 
            [np.zeros( (1, 4) )]
        ])
