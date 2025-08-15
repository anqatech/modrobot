import numpy as np
from .rigid_body_representation import RigidBodyRepresentation

class RigidBodyWrench:
    __slots__ = (
        "_body_wrench",
        "_representation",
        "_space_wrench",
    )

    def __init__(self, wrench_vector, representation, wrench_type):
        if not isinstance(wrench_vector, np.ndarray) or wrench_vector.shape != (6, 1):
            raise TypeError("The rigid body wrench must be a 6x1 NumPy array.")
        if not isinstance(representation, RigidBodyRepresentation):
            raise TypeError("The rigid body representation must be of type RigidBodyRepresentation.")
    
        self._representation = representation
        wrench_type = wrench_type.lower()

        if wrench_type == "body":
            self._body_wrench = wrench_vector
            self._space_wrench = self.representation.adjoint_representation_inverse.T @ self.body_wrench
        elif wrench_type == "space":
            self._space_wrench = wrench_vector
            self._body_wrench = self.representation.adjoint_representation.T @ self.space_wrench
        else:
            raise ValueError("wrench_type must be either 'body' or 'space'.")

    @property
    def body_wrench(self):
        return self._body_wrench

    @property
    def space_wrench(self):
        return self._space_wrench
    
    @property
    def representation(self):
        return self._representation
    
    def __repr__(self):
        body_wrench_str = np.array2string(
            self.body_wrench,
            separator=", ",
            precision=16,
            suppress_small=False,
        )

        return f"RigidBodyWrench(np.array({body_wrench_str}), wrench_type='body')"

    def __str__(self):
        body_wrench_str = np.array2string(
            self.body_wrench,
            precision=4,
            suppress_small=True,
            separator=" ",
        )
        space_wrench_str = np.array2string(
            self.space_wrench,
            precision=4,
            suppress_small=True,
            separator=" ",
        )

        return f"Body Wrench:\n\n{body_wrench_str}\n\nSpace Wrench:\n\n{space_wrench_str}"
