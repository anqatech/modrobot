import numpy as np
from .rigid_body_representation import RigidBodyRepresentation


class RigidBodyTwist:
    __slots__ = (
        "_w",
        "_v",
        "_representation",
        "_frame",
        "_twist_type",
        "_space_twist",
        "_body_twist",
    )

    def __init__(self, w, v, representation, frame, twist_type):
        self._w = w
        self._v = v
        self._representation = representation
        self._frame = frame
        self._twist_type = twist_type

        w_norm = np.linalg.norm(self.w)
        v_norm = np.linalg.norm(self.v)

        if self.twist_type == "screw":
            if np.isclose(w_norm, 0.0, atol=1e-12):
                if not np.isclose(v_norm, 1.0, atol=1e-12):
                    raise ValueError("The angular velocity w and velocity v entered do not correspond to a twist_type='screw'")
            elif not np.isclose(w_norm, 1.0, atol=1e-12):
                raise ValueError("The angular velocity w and velocity v entered do not correspond to a twist_type='screw'")
        elif self.twist_type == "twist":
            pass
        else:
            raise ValueError("The twist_type input can only be one of two values, either 'screw' or 'twist'")

        if self.frame not in ("space", "body"):
            raise ValueError("The frame input must be 'space' or 'body'")
    
        if self.frame == "space":
            self._space_twist = np.concatenate((w, v), axis=0)
            self._body_twist = self.representation.adjoint_representation_inverse @ self.space_twist
        if self.frame == "body":
            self._body_twist = np.concatenate((w, v), axis=0)
            self._space_twist = self.representation.adjoint_representation @ self.body_twist

    @classmethod
    def creating_screw_axis_from_w_q_h(cls, w, q, h, representation, frame, twist_type):
        if twist_type != "screw":
            raise ValueError("twist_type must be 'screw'")
        if not isinstance(w, np.ndarray) or w.shape != (3, 1):
            raise TypeError("w must be a 3x1 NumPy array.")
        if not isinstance(q, np.ndarray) or q.shape != (3, 1):
            raise TypeError("q must be a 3x1 NumPy array.")
        if not isinstance(representation, RigidBodyRepresentation):
            raise TypeError("The rigid body representation must be of type RigidBodyRepresentation.")

        w_norm = np.linalg.norm(w)

        if np.isclose(w_norm, 1.0, atol=1e-12):
            v = -np.linalg.cross(w.squeeze(), q.squeeze()).reshape((3, 1))
        elif np.isclose(w_norm, 0.0, atol=1e-12):
            v = np.array([[1.0], [0.0], [0.0]])
        else:
            raise ValueError("w must be a unit vector or the zero vector")
        
        return cls(w, v, representation, frame, twist_type)

    @property
    def w(self):
        return self._w

    @property
    def v(self):
        return self._v

    @property
    def representation(self):
        return self._representation

    @property
    def frame(self):
        return self._frame

    @property
    def twist_type(self):
        return self._twist_type

    @property
    def space_twist(self):
        return self._space_twist
    
    @property
    def space_twist_matrix(self):
        return self.space_twist_skew_matrix()
    
    @property
    def body_twist(self):
        return self._body_twist

    @property
    def body_twist_matrix(self):
        return self.body_twist_skew_matrix()

    def __str__(self):
        space_twist_str = np.array2string(
            self.space_twist,
            precision=4,
            suppress_small=True,
            separator=" ",
        )
        body_twist_str = np.array2string(
            self.body_twist,
            precision=4,
            suppress_small=True,
            separator=" ",
        )

        return f"Space Twist:\n\n{space_twist_str}\n\nBody Twist:\n\n{body_twist_str}"

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
