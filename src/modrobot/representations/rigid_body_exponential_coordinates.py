import numpy as np
from .rigid_body_representation import RigidBodyRepresentation
from .rotation_matrix import RotationMatrix
from modrobot.utilities.skew_utils import skew_matrix


class RigidBodyExponentialCoordinates:
    __slots__ = (
        "_exponential_coordinates",
        "_theta",
        "_screw_axis",
        "_representation",
    )

    def __init__(self, exponential_coordinates):
        if not isinstance(exponential_coordinates, np.ndarray) or exponential_coordinates.shape != (6, 1):
            raise TypeError("The exponential coordinates must be a 6x1 NumPy array.")
            
        self._exponential_coordinates = exponential_coordinates

        w_theta = self.exponential_coordinates[0:3]
        v_theta = self.exponential_coordinates[3:]
        
        theta = np.linalg.norm(w_theta)

        if np.isclose(theta, 0.0, atol=1e-12):
            R = np.eye(3)
            p = v_theta
            self._theta = np.linalg.norm(v_theta)
            if np.isclose(self.theta, 0.0, atol=1e-12):
                 self._screw_axis = np.zeros((6, 1))
            else:
                 self._screw_axis = np.vstack([np.zeros((3, 1)), v_theta / self.theta])
        else:
            self._theta = theta
            self._screw_axis = self.exponential_coordinates / self.theta
            
            w = w_theta / theta
            v = v_theta / theta
            
            R = RotationMatrix.from_exponential_coordinates(w_theta).rotation_matrix

            w_skew = skew_matrix(w)
            G_theta = np.eye(3) * self.theta + (1 - np.cos(self.theta)) * w_skew + (self.theta - np.sin(self.theta)) * (w_skew @ w_skew)
            p = G_theta @ v

        self._representation = RigidBodyRepresentation(R, p, check=False)

    @property
    def exponential_coordinates(self):
        return self._exponential_coordinates

    @property
    def theta(self):
        return self._theta

    @property
    def screw_axis(self):
        return self._screw_axis

    @property
    def representation(self):
        return self._representation

    def __str__(self):
        exponential_coordinates_str = np.array2string(
            self.exponential_coordinates, 
            separator=" ",
            precision=4, 
            suppress_small=True, 
        )
        return f"Screw Motion (Exponential Coords):\n{exponential_coordinates_str}"
    
