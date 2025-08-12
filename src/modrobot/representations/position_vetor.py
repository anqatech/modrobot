import numpy as np

class PositionVector:
    __slots__ = ("_position_vector")

    def __init__(self, position_vector):
        if not isinstance(position_vector, np.ndarray) or position_vector.shape != (3, 1):
            raise TypeError("Input must be a 3x1 NumPy array.")
        
        self._position_vector = position_vector

    @property
    def vector(self):
        return self._position_vector

    def __repr__(self):
        position_vector_str = np.array2string(
            self._position_vector,
            separator=", ",
            precision=16,
            suppress_small=False,
        )

        return f"PositionVector(np.array({position_vector_str}))"

    def __str__(self):
        position_vector_str = np.array2string(
            self._position_vector,
            precision=4,
            suppress_small=True,
            separator=" ",
        )

        return f"Position Vector:\n\n{position_vector_str}"
