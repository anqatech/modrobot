import numpy as np
from .rigid_body_representation import RigidBodyRepresentation
from .rotation_matrix import RotationMatrix
from .position_vector import PositionVector

class RigidBodyScrew:
    __slots__ = (
        "_exponential_coordinates", 
        "_representation", 
        "_screw_axis", 
        "_theta"
    )

    def __init__(self, exponential_coordinates, representation):
        self._exponential_coordinates = exponential_coordinates
        self._representation = representation
        self._compute_axis_and_theta()

    @classmethod
    def from_exponential_coordinates(cls, exponential_coordinates, check=True):
        if not isinstance(exponential_coordinates, np.ndarray) or exponential_coordinates.shape != (6, 1):
            raise TypeError("The exponential coordinates must be a 6x1 NumPy array.")

        omega_theta = exponential_coordinates[0:3]
        v_theta = exponential_coordinates[3:]
        
        theta = np.linalg.norm(omega_theta)

        if np.isclose(theta, 0.0, atol=1e-12):
            R = np.eye(3)
            p = v_theta
        else:
            omega = omega_theta / theta
            v = v_theta / theta
            
            R = RotationMatrix.from_exponential_coordinates(
                omega_theta, 
                check=check
            ).rotation_matrix

            w_skew = cls.skew_matrix(omega)
            G_theta = np.eye(3) * theta + (1 - np.cos(theta)) * w_skew + (theta - np.sin(theta)) * (w_skew @ w_skew)
            p = G_theta @ v
            
        representation = RigidBodyRepresentation(R, p, check=False)
        return cls(exponential_coordinates, representation)

    @classmethod
    def from_transformation(cls, representation):
        if not isinstance(representation, RigidBodyRepresentation):
            raise TypeError("Input must be a RigidBodyRepresentation object.")

        R = representation.rotation_matrix
        p = representation.origin_position
        
        rot_obj = RotationMatrix(R, check=False)
        omega_theta = rot_obj.exponential_coordinates
        theta = np.linalg.norm(omega_theta)
        
        if np.isclose(theta, 0.0, atol=1e-12):
            omega_theta = np.zeros((3, 1))
            v_theta = p
        else:
            omega = omega_theta / theta
            w_skew = RotationMatrix.skew_matrix(omega)
            # Using textbook formula (3.92) for G_inv
            G_inv_theta = (np.eye(3) / theta) - 0.5 * w_skew + \
                          ((1.0 / theta) - 0.5 / np.tan(theta / 2.0)) * (w_skew @ w_skew)
            v = G_inv_theta @ p
            v_theta = v * theta
            
        exp_coords = np.vstack([omega_theta, v_theta])
        return cls(exp_coords, representation)

    @property
    def exponential_coordinates(self):
        return self._exponential_coordinates

    @property
    def representation(self):
        return self._representation

    @property
    def screw_axis(self):
        return self._screw_axis

    @property
    def theta(self):
        return self._theta

    def _compute_axis_and_theta(self):
        omega_theta = self._exponential_coordinates[0:3]
        v_theta = self._exponential_coordinates[3:]
        
        theta = np.linalg.norm(omega_theta)
        
        if np.isclose(theta, 0.0, atol=1e-12):
            self._theta = np.linalg.norm(v_theta)
            if np.isclose(self._theta, 0.0, atol=1e-12):
                 self._screw_axis = np.zeros((6, 1))
            else:
                 self._screw_axis = np.vstack([np.zeros((3, 1)), v_theta / self._theta])
        else:
            self._theta = theta
            self._screw_axis = self._exponential_coordinates / self._theta

    def __repr__(self):
        exponential_coordinates_str = np.array2string(
            self.exponential_coordinates, 
            separator=", ", 
            precision=16, 
            suppress_small=False
        )
        return f"RigidBodyScrew.from_exponential_coordinates(np.array({exponential_coordinates_str}))"

    def __str__(self):
        exponential_coordinates_str = np.array2string(
            self.exponential_coordinates, 
            separator=" ",
            precision=4, 
            suppress_small=True, 
        )
        return f"Screw Motion (Exponential Coords):\n{exponential_coordinates_str}"

    @staticmethod
    def skew_matrix(vector):
        v1, v2, v3 = vector.squeeze()
    
        return np.array([
            [0, -v3, v2],
            [v3, 0, -v1],
            [-v2, v1, 0],
        ])
    
