import numpy as np
import blockbeamParam as P

class ctrlPID:
    def __init__(self, flag=True):
        self.flag = flag
        self.z_e = P.length / 2

        # inner loop
        zeta_theta = 0.707
        t_r_theta = .12
        w_n_theta = 0.5 * np.pi / t_r_theta / np.sqrt(1 - zeta_theta**2)
        b = 1.0/3.0 * P.m2 * P.length**2 + P.m1 * self.z_e**2

        self.kd_theta = 2 * zeta_theta * w_n_theta * b / P.length
        self.kp_theta = w_n_theta**2 * b / P.length
        self.ki_theta = 0.

        # PD gains
        print('kp_theta: ', self.kp_theta)
        print('kd_theta: ', self.kd_theta)

        # outer loop
        zeta_z = 0.707
        t_r_z = 1.2
        w_n_z = 0.5 * np.pi / t_r_z / np.sqrt(1 - zeta_z**2)

        self.kd_z = - 2 * zeta_z * w_n_z / P.g
        self.kp_z = - w_n_z**2 / P.g
        self.ki_z = -0.1
        # PD gains
        print('kp_z: ', self.kp_z)
        print('kd_z: ', self.kd_z)

        self.prev_e_theta = 0.0
        self.prev_e_z = 0.0

        self.prev_z = 0.0
        self.prev_theta = 0.0

        self.e_z_dot = 0.0
        self.e_theta_dot = 0.0

        self.z_dot = 0.0        
        self.theta_dot = 0.0

        self.integrator_theta = 0.0
        self.integrator_z = 0.0

        self.sigma_theta = 0.05
        self.sigma_z = 0.05

    def update(self, z_r, state):
        z = state[0][0]
        theta = state[1][0]

        e_z = z_r - z
        self.z_dot = (2*self.sigma_z - P.Ts) / (2*self.sigma_z + P.Ts) * self.z_dot + 2. / (2 * self.sigma_z + P.Ts) * (z - self.prev_z)
        if np.abs(self.z_dot) <= 0.1:
            self.integrator_z += (P.Ts / 2) * (e_z + self.prev_e_z)

        if self.flag:
            theta_r = self.kp_z * e_z - self.kd_z * self.z_dot + self.ki_z * self.integrator_z
        else:
            self.e_z_dot = (2*self.sigma_z - P.Ts) / (2*self.sigma_z + P.Ts) * self.e_z_dot + 2. / (2 * self.sigma_z + P.Ts) * (e_z - self.prev_e_z)
            theta_r = self.kp_z * e_z + self.kd_z * self.e_z_dot + self.ki_z * self.integrator_z

        self.prev_z = z
        self.prev_e_z = e_z

        e_theta = theta_r - theta
        self.integrator_theta += (P.Ts / 2) * (e_theta + self.prev_e_theta)

        if self.flag:
            self.theta_dot = (2*self.sigma_theta - P.Ts) / (2*self.sigma_theta + P.Ts) * self.theta_dot + 2. / (2 * self.sigma_theta + P.Ts) * (theta - self.prev_theta)
            f_tilde = self.kp_theta * e_theta - self.kd_theta * self.theta_dot + self.ki_theta * self.integrator_theta
        else:
            self.e_theta_dot = (2*self.sigma_theta - P.Ts) / (2*self.sigma_theta + P.Ts) * self.e_theta_dot + 2. / (2 * self.sigma_theta + P.Ts) * (e_theta - self.prev_e_theta)
            f_tilde = self.kp_theta * e_theta + self.kd_theta * self.e_theta_dot + self.ki_theta * self.integrator_theta

        self.prev_theta = theta
        self.prev_e_theta = e_theta

        f_fl = (P.m1 * P.g * z + P.length / 2 * P.m2 * P.g) / P.length
        f = f_tilde + f_fl

        return saturate(f, P.F_max)

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u