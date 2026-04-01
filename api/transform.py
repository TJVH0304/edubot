'''
Custom class to define transformation matrices. Features multiple constructors in the form of classmethods. Allowed constructors:

- fromMatrix: provide it the entire 4x4 matrix
- fromRT: provide it the 3x3 rotation matrix and the 3x1 translation vector
- identity: simply 4x4 identity matrix
- fromTranslation: only a 3x1 translation
- fromRotation: only a 3x3 rotation
- fromEuler: provide (up to) 3 euler angles, and the translation vector. order ZYX

Rotation matrix can be accessed using .R(), and the translation vector using .t(). Inverse accessed using .inverse()
'''

import numpy as np


class Transformation:
    def __init__(self, M):
        M = np.asarray(M, dtype=float)
        if M.shape != (4, 4):
            raise ValueError('Transformation matrix must be 4x4')
        self.M = M.copy()

    def R(self):
        return self.M[0:3, 0:3]

    def t(self):
        return self.M[0:3, 3]

    def inverse(self):
        R = self.R()
        t = self.t()
        R_inv = R.T
        t_inv = -R_inv @ t
        return Transformation.fromRT(R_inv, t_inv)

    def apply(self, p):
        p = np.asarray(p, dtype=float)

        if p.shape == (3,):
            return self.R() @ p + self.t()

        if p.ndim == 2 and p.shape[1] == 3:
            return (p @ self.R().T) + self.t()

        raise ValueError("Point must have shape (3,) or (N, 3)")

    def __matmul__(self, other):
        if isinstance(other, Transformation):
            return Transformation(self.M @ other.M)
        return self.apply(other)

    def __mul__(self, other):
        return self.__matmul__(other)

    def __repr__(self):
        return f'Transformation'

    @classmethod
    def fromMatrix(cls, M):
        return cls(M)

    @classmethod
    def fromRT(cls, R, T):
        R = np.asarray(R, dtype=float)
        T = np.asarray(T, dtype=float)

        if R.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        if T.shape != (3,):
            raise ValueError("Translation vector must have shape (3,)")

        M = np.identity(4)
        M[0:3, 0:3] = R
        M[0:3, 3] = T

        return cls(M)

    @classmethod
    def identity(cls):
        return cls(np.identity(4))

    @classmethod
    def fromTranslation(cls, T):
        return cls.fromRT(np.identity(3), T)

    @classmethod
    def fromRotation(cls, R):
        return cls.fromRT(R, np.zeros(3))

    @classmethod
    def fromEuler(cls, yaw=0, pitch=0, roll=0, T=np.zeros(3)):
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        Rz = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ])

        Ry = np.array([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ])

        R = Rz @ Ry @ Rx
        return cls.fromRT(R, T)
