import numpy as np
from .rigid_body_representation import RigidBodyRepresentation


class RigidBodyTwist:
    __slots__ = (
        "_space_twist",
        "_body_twist",
        "_representation",
    )

    def __init__(self, space_twist, body_twist, representation):
        """Private constructor. Use classmethods to create instances."""
        self._space_twist = space_twist
        self._body_twist = body_twist
        self._representation = representation

    @classmethod
    def from_body_twist(cls, body_twist_vector, representation):
        if not isinstance(body_twist_vector, np.ndarray) or body_twist_vector.shape != (6, 1):
            raise TypeError("The body twist must be a 6x1 NumPy array.")
        if not isinstance(representation, RigidBodyRepresentation):
            raise TypeError("The rigid body representation must be of type RigidBodyRepresentation.")

        space_twist_vector = representation.adjoint_representation @ body_twist_vector
        return cls(space_twist_vector, body_twist_vector, representation)

    @classmethod
    def from_space_twist(cls, space_twist_vector, representation):
        if not isinstance(space_twist_vector, np.ndarray) or space_twist_vector.shape != (6, 1):
            raise TypeError("The space twist must be a 6-element NumPy array.")
        if not isinstance(representation, RigidBodyRepresentation):
            raise TypeError("The rigid body representation must be of type RigidBodyRepresentation.")

        body_twist_vector = representation.adjoint_representation_inverse @ space_twist_vector
        return cls(space_twist_vector, body_twist_vector, representation)

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
