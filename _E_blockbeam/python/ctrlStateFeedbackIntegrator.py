import numpy as np
import control as cnt
import blockbeamParam as P


class ctrlStateFeedbackIntegrator:
    def __init__(self):
        #--------------------------------------------------
        # State Feedback Control Design
        #--------------------------------------------------
        # tuning parameters
        tr_z = 1.2  # rise time for position
        tr_theta = 0.25  # rise time for angle
        zeta_z = 0.707  # damping ratio position
        zeta_th = 0.707  # damping ratio angle
        wn_th = 2.2 / tr_theta  # natural frequency for angle
        integrator_pole = -1.0
        # Augmented State Space Equations
        # xdot = A*x + B*u
        # y = C*x
        self.z_e = P.length / 2.0
        b = P.m1 * (self.z_e**2) + 1./3. * P.m2 * (P.length**2)
        A = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -P.g, 0.0, 0.0],
            [-(P.m1 * P.g) / b, 0.0, 0.0, 0.0],
            ])
        
        B = np.array([[0.0],
                      [0.0],
                      [0.0],
                      [P.length / b]])
        C = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
        Cr = np.array([[1.0, 0.0, 0.0, 0.0]])
        # form augmented system
        A1 = np.vstack((
                np.hstack((A, np.zeros((4,1)))),
                np.hstack((-Cr, np.zeros((1,1))))))
        B1 = np.vstack((B, np.zeros((1,1))))
        # gain calculation
        wn_z = 2.2 / tr_z  # natural frequency for position
        des_char_poly = np.convolve(
            np.convolve([1, 2 * zeta_z * wn_z, wn_z**2],
                        [1, 2 * zeta_th * wn_th, wn_th**2]),
            np.poly([integrator_pole]))
        des_poles = np.roots(des_char_poly)
        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A1, B1)) != 5:
            print("The system is not controllable")
        else:
            K1 = cnt.place(A1, B1, des_poles)
            self.K = K1[0][0:4]
            self.ki = K1[0][4]
        # print gains to terminal
        print('K: ', self.K)
        print('ki: ', self.ki)
        #--------------------------------------------------
        # variables to implement integrator
        self.integrator_z = 0.0  # integrator
        self.error_z_d1 = 0.0  # error signal delayed by 1 sample
    
    def update(self, z_r, x, d):
        z = x[0][0]

        e = z_r - z
        self.integrator_z = self.integrator_z + (P.Ts / 2.0) * (e + self.error_z_d1)

        # Compute the state feedback controller
        F_fl = P.m1*P.g*(self.z_e/P.length) + P.m2*P.g/2.0
        F_unsat = - self.K @ (x - [[self.z_e], [0], [0], [0]]) - self.ki * self.integrator_z
        F = saturate(F_unsat[0] + F_fl + d, P.F_max)

        self.error_z_d1 = e

        return F


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u

