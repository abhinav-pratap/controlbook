import numpy as np
import blockbeamParam as P

class ctrlPD:
    def __init__(self):
        # Declare pole locations
        zeta = 0.707
        t_r = 15
        w_n = 2.2 / t_r


        self.kd = - 2 * zeta * w_n / P.g
        self.kp = - w_n**2 / P.g
        self.z_e = P.length / 2
        # PD gains
        print('kp: ', self.kp)
        print('kd: ', self.kd)

    def update(self, z_r, state):
        zdot = state[2][0]
        thetadot = state[3][0]

        z = state[0][0]
        theta = state[1][0]

        f_tilde = self.kp * (z_r - z) - self.kd * zdot
        # f_tilde = 0
        f_fl = (P.m1 * P.g * z + P.length / 2 * P.m2 * P.g) / P.length

        f = f_tilde + f_fl

        return saturate(f, P.F_max)

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u