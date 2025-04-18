import numpy as np
import control as cnt
import hummingbirdParam as P
import sys

class ctrlStateFeedbackIntegrator:
    def __init__(self):
        #--------------------------------------------------
        # State Feedback Control Design
        #--------------------------------------------------
        # longitudinal controller
        tr_th = 0.6
        wn_th = 2.2 / tr_th
        zeta_th = 0.707
        pi_th = -wn_th/2.0

        b_theta = P.ellT/(P.m1 * P.ell1**2 + P.m2 * P.ell2**2 + P.J1y + P.J2y)
        A_lon = np.array([[0.0, 1.0],
                      [0.0, 0.0]])
        B_lon = np.array([[0.0],
                      [b_theta]])        
        Cr_lon = np.array([[1.0, 0.0]])
        # form augmented system
        A1_lon = np.vstack((np.hstack((A_lon, np.zeros((np.size(A_lon,1),1)))), 
                        np.hstack((-Cr_lon, np.array([[0.0]]))) ))
        B1_lon = np.vstack( (B_lon, 0.0) )
        des_char_poly_lon = np.convolve([1, 2 * zeta_th * wn_th, wn_th**2], 
                                    [1, -pi_th])
        des_poles_lon = np.roots(des_char_poly_lon)
        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A1_lon, B1_lon)) != 3:
            print("The longitudinal system is not controllable")
            sys.exit(1)
        else:
            K1_lon = cnt.place(A1_lon, B1_lon, des_poles_lon)
            self.k_lon = K1_lon[0][0:2]
            self.k_th = K1_lon[0][0]
            self.k_thdot = K1_lon[0][1]
            self.ki_lon = K1_lon[0][2]

        # lateral controller
        tr_psi = 2.0
        zeta_psi = 0.9
        wn_psi = 0.5 * np.pi / tr_psi / np.sqrt(1 - zeta_psi**2)

        M = 6.0

        tr_phi = tr_psi / M
        zeta_phi = 0.9
        wn_phi = 0.5 * np.pi / tr_phi / np.sqrt(1 - zeta_phi**2)        
        pi_psi =  -wn_psi/2.0

        f_e = (P.m1*P.ell1 + P.m2*P.ell2)*P.g/P.ellT
        J_T = P.m1 * P.ell1**2 + P.m2 * P.ell2**2 + P.J2z + P.m3 * (P.ell3x**2 + P.ell3y**2)
        b_psi = P.ellT * f_e / (J_T + P.J1z)

        A_lat = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [b_psi, 0.0, 0.0, 0.0],
            ])
        
        B_lat = np.array([[0.0],
                      [0.0],
                      [1.0 / P.J1x],
                      [0.0]])
        C_lat = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
        Cr_lat = np.array([[0.0, 1.0, 0.0, 0.0]])
        # form augmented system
        A1_lat = np.vstack((
                np.hstack((A_lat, np.zeros((4,1)))),
                np.hstack((-Cr_lat, np.zeros((1,1))))))
        B1_lat = np.vstack((B_lat, np.zeros((1,1))))
        des_char_poly_lat = np.convolve(
            np.convolve([1, 2 * zeta_phi * wn_phi, wn_phi**2],
                        [1, 2 * zeta_psi * wn_psi, wn_psi**2]),
                        np.poly([pi_psi]))
        des_poles_lat = np.roots(des_char_poly_lat)
        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A1_lat, B1_lat)) != 5:
            print("The lateral system is not controllable")
        else:
            K1_lat = cnt.place(A1_lat, B1_lat, des_poles_lat)
            self.k_lat = K1_lat[0][0:4]
            self.k_phi = self.k_lat[0]
            self.k_psi = self.k_lat[1]
            self.k_phidot = self.k_lat[2]
            self.k_psidot = self.k_lat[3]
            self.ki_lat = K1_lat[0][4]

        # print gains to terminal
        print('K_lon: [', self.k_th, ',', self.k_thdot, ']')
        print('ki_lon: ', self.ki_lon)         
        print('K_lat: [', self.k_phi, ',', self.k_psi, ',', self.k_phidot, ',', self.k_psidot, ']')
        print('ki_lat: ', self.ki_lat)        
        #--------------------------------------------------
        # saturation limits
        theta_max = 30.0 * np.pi / 180.0  # Max theta, rads
        #--------------------------------------------------
        self.Ts = P.Ts
        sigma = 0.05  # cutoff freq for dirty derivative
        self.beta = (2 * sigma - self.Ts) / (2 * sigma + self.Ts)
        self.phi_d1 = 0.
        self.phi_dot = 0.
        self.theta_d1 = 0.
        self.theta_dot = 0.
        self.psi_d1 = 0.
        self.psi_dot = 0.        
        # variables to implement integrator
        self.integrator_th = 0.0  
        self.error_th_d1 = 0.0  
        self.integrator_psi = 0.0  
        self.error_psi_d1 = 0.0 

    def update(self, r: np.ndarray, y: np.ndarray, d_lon: float, d_lat: float):
        theta_ref = r[0][0]
        psi_ref = r[1][0]
        phi = y[0][0]
        theta = y[1][0]
        psi = y[2][0]
        force_equilibrium = P.g * (P.m1 * P.ell1 + P.m2 * P.ell2) * np.cos(theta) / P.ellT
        # update differentiators
        self.phi_dot = self.beta * self.phi_dot \
                       + (1 - self.beta) * ((phi - self.phi_d1) / self.Ts)
        self.phi_d1 = phi
        self.theta_dot = self.beta * self.theta_dot \
                       + (1 - self.beta) * ((theta - self.theta_d1) / self.Ts)
        self.theta_d1 = theta
        self.psi_dot = self.beta * self.psi_dot \
                       + (1 - self.beta) * ((psi - self.psi_d1) / self.Ts)
        self.psi_d1 = psi
        # integrate error
        error_th = theta_ref - theta
        error_psi = psi_ref - psi
        self.integrator_th = self.integrator_th + (P.Ts / 2.0) * (error_th + self.error_th_d1)
        self.integrator_psi = self.integrator_psi + (P.Ts / 2.0) * (error_psi + self.error_psi_d1)
        self.error_th_d1 = error_th
        self.error_psi_d1 = error_psi

        x_lon = np.array([[theta], [self.theta_dot]])
        x_lat = np.array([[phi], [psi], [self.phi_dot], [self.psi_dot]])

        # longitudinal control
        force_unsat = force_equilibrium - self.k_lon @ (x_lon - [[0], [0]]) - self.ki_lon * self.integrator_th
        force = saturate(force_unsat[0] + d_lon, -P.force_max, P.force_max)
        # lateral control
        torque_unsat = - self.k_lat @ (x_lat - [[0], [0], [0], [0]]) - self.ki_lat * self.integrator_psi
        torque = saturate(torque_unsat[0] + d_lat, -P.torque_max, P.torque_max)
        # convert force and torque to pwm signals
        pwm = np.array([[force + torque / P.d],               # u_left
                      [force - torque / P.d]]) / (2 * P.km)   # r_right          
        pwm = saturate(pwm, 0, 1)
        return pwm, np.array([[0], [theta_ref], [psi_ref]])


def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        if u > up_limit:
            u = up_limit
        if u < low_limit:
            u = low_limit
    else:
        for i in range(0, u.shape[0]):
            if u[i][0] > up_limit:
                u[i][0] = up_limit
            if u[i][0] < low_limit:
                u[i][0] = low_limit
    return u
