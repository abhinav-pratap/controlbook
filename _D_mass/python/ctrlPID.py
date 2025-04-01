import numpy as np
import massParam as P

class ctrlPID:
    def __init__(self, flag=True):
        self.flag = flag

        # Declare pole locations
        zeta = 0.7
        t_r = 1.85
        w_n = 0.5 * np.pi / t_r / np.sqrt(1.0 - zeta**2)


        self.kd = (2.0 * zeta * w_n) * P.m - P.b
        self.kp = w_n**2 * P.m - P.k
        self.ki = 1.9

        # PD gains
        print('kp: ', self.kp)
        print('kd: ', self.kd)

        self.sigma = 0.05
        self.e_dot = 0.0
        self.z_dot = 0.0
        self.e_prev = 0.0
        self.z_prev = 0.0
        self.integrator = 0.0

    def update(self, z_r, x):
        z = x[0][0]
        e = z_r - z
        self.integrator += (P.Ts / 2) * (e + self.e_prev)

        if self.flag:
            self.z_dot = (2*self.sigma - P.Ts) / (2*self.sigma + P.Ts) * self.z_dot + 2. / (2 * self.sigma + P.Ts) * (z - self.z_prev)
            z_tilde_unsat = self.kp * e - self.kd * self.z_dot + self.ki * self.integrator
        else:
            self.e_dot = (2*self.sigma - P.Ts) / (2*self.sigma + P.Ts) * self.e_dot + 2. / (2 * self.sigma + P.Ts) * (e - self.e_prev)
            z_tilde_unsat = self.kp * e + self.kd * self.e_dot + self.ki * self.integrator

        z_tilde_sat = saturate(z_tilde_unsat, P.F_max)

        if self.ki != 0.0:
            self.integrator = self.integrator \
            + 1.0 / self.ki * (z_tilde_sat - z_tilde_unsat)

        self.e_prev = e
        self.z_prev = z

        return z_tilde_sat

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u