import numpy as np
import massParam as P

class ctrlPD:
    def __init__(self):
        # Declare pole locations
        zeta = 0.707
        t_r = 2.0
        w_n = 2.2 / t_r


        self.kd = (2.0 * zeta * w_n) * P.m - P.b
        self.kp = w_n**2 * P.m - P.k

        # PD gains
        print('kp: ', self.kp)
        print('kd: ', self.kd)

    def update(self, z_r, x):
        z = x[0][0]
        zdot = x[1][0]

        z_tilde = self.kp * (z_r - z) - self.kd * zdot

        return saturate(z_tilde, P.F_max)

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u