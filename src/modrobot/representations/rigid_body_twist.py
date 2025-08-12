import numpy as np

class RigidBodyTwist:
    __slots__ = ("_body_twist")

    def __init__(self, body_twist):
        if not isinstance(body_twist, np.ndarray) or body_twist.shape != (6, 1):
            raise TypeError("The body twist must be a 6x1 NumPy array.")
            
        self._body_twist = body_twist

    @property
    def body_twist(self):
        return self._body_twist

    @property
    def body_twist_matrix(self):
        return self.body_twist_skew_matrix()

    def __repr__(self):
        body_twist_str = np.array2string(
            self.body_twist,
            separator=", ",
            precision=16,
            suppress_small=False,
        )

        return f"RigidBodyTwist(np.array({body_twist_str}))"

    def __str__(self):
        body_twist_str = np.array2string(
            self.body_twist,
            precision=4,
            suppress_small=True,
            separator=" ",
        )

        return f"Body Twist:\n\n{body_twist_str}"

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

    def body_twist_skew_matrix(self):
        w = self.body_twist[0:3]
        v = self.body_twist[3:]
    
        return np.block([ 
            [self.skew_matrix(w), v] , 
            [np.zeros( (1, 4) )]
        ])
