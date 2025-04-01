import numpy as np
import blockbeamParam as P

class ctrlPD:
    def __init__(self):
        self.z_e = P.length / 2

        # inner loop
        zeta_theta = 0.707
        t_r_theta = .3
        w_n_theta = 0.5 * np.pi / t_r_theta / np.sqrt(1 - zeta_theta**2)
        b = 1.0/3.0 * P.m2 * P.length**2 + P.m1 * self.z_e**2

        self.kd_theta = 2 * zeta_theta * w_n_theta * b / P.length
        self.kp_theta = w_n_theta**2 * b / P.length

        # PD gains
        print('kp_theta: ', self.kp_theta)
        print('kd_theta: ', self.kd_theta)

        # outer loop
        zeta_z = 0.707
        t_r_z = 3
        w_n_z = 0.5 * np.pi / t_r_z / np.sqrt(1 - zeta_z**2)

        self.kd_z = - 2 * zeta_z * w_n_z / P.g
        self.kp_z = - w_n_z**2 / P.g
        # PD gains
        print('kp_z: ', self.kp_z)
        print('kd_z: ', self.kd_z)

    def update(self, z_r, state):
        zdot = state[2][0]
        thetadot = state[3][0]

        z = state[0][0]
        theta = state[1][0]

        theta_r = self.kp_z * (z_r - z) - self.kd_z * zdot
        f_tilde = self.kp_theta * (theta_r - theta) - self.kd_theta * thetadot

        f_fl = (P.m1 * P.g * z + P.length / 2 * P.m2 * P.g) / P.length

        f = f_tilde + f_fl

        return saturate(f, P.F_max)

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u