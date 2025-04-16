import numpy as np
import control as cnt
import blockbeamParam as P

class ctrlStateFeedback:
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
        
        # State Space Equations
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
        
        # gain calculation
        wn_z = 2.2 / tr_z  # natural frequency for position
        # des_char_poly = np.array([1, 0, 2*wn_z*zeta_z, 0, wn_z**2])
        des_char_poly = np.convolve([1, 2 * zeta_z * wn_z, wn_z**2],
                                    [1, 2 * zeta_th * wn_th, wn_th**2])
        print(des_char_poly)
        des_poles = np.roots(des_char_poly)

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A, B)) != 4:
            print("The system is not controllable")
        else:
            self.K = cnt.place(A, B, des_poles)
            Cr = np.array([[1.0, 0.0, 0.0, 0.0]])
            self.kr = -1.0 / (Cr @ np.linalg.inv(A-B @ self.K) @ B)
        # print gains to terminal
        print('K: ', self.K)
        print('kr: ', self.kr)

    def update(self, z_r, x):
        z = x[0][0]
        # Compute the state feedback controller
        F_fl = P.m1*P.g*(self.z_e/P.length) + P.m2*P.g/2.0
        F_unsat = - self.K @ (x - [[self.z_e], [0], [0], [0]]) + self.kr * (z_r - self.z_e)
        F = saturate(F_unsat[0][0] + F_fl, P.F_max)
        return F


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u

