import numpy as np
import hummingbirdParam as P


class ctrlPD:
    def __init__(self):
        ######
        # Longitudinal controller gains
        ######
        # tuning parameters
        tr_pitch = 1.0
        zeta_pitch = 0.707
        # self.ki_pitch = 
        # gain calculation
        b_theta = P.ellT/(P.m1 * P.ell1**2 + P.m2 * P.ell2**2 + P.J1y + P.J2y)
        #print('b_theta: ', b_theta)
        wn_pitch = 0.5 * np.pi / tr_pitch / np.sqrt(1 - zeta_pitch**2)
        self.kp_pitch = wn_pitch**2 / b_theta
        self.kd_pitch = 2 * zeta_pitch * wn_pitch / b_theta
        # print gains to terminal
        print('kp_pitch: ', self.kp_pitch)
        # print('ki_pitch: ', self.ki_pitch)
        print('kd_pitch: ', self.kd_pitch) 
        # sample rate of the controller
        self.Ts = P.Ts
        # dirty derivative parameters
        sigma = 0.05  # cutoff freq for dirty derivative
        self.beta = (2 * sigma - self.Ts) / (2 * sigma + self.Ts)
        # delayed variables
        self.theta_d1 = 0.
        self.theta_dot = 0.
        self.integrator_theta = 0.
        self.error_theta_d1 = 0.  # pitch error delayed by 1

        ######
        # Lateral controller gains 
        ######
        tr_roll = 0.9
        zeta_roll = 0.707
        wn_roll = 0.5 * np.pi / tr_roll / np.sqrt(1 - zeta_roll**2)
        self.kp_roll = wn_roll**2 * P.J1x
        self.kd_roll = 2 * zeta_roll * wn_roll * P.J1x

        print('kp_roll: ', self.kp_roll)
        print('kd_roll: ', self.kd_roll) 
        
        f_e = (P.m1*P.ell1 + P.m2*P.ell2)*P.g/P.ellT
        J_T = P.m1 * P.ell1**2 + P.m2 * P.ell2**2 + P.J2z + P.m3 * (P.ell3x**2 + P.ell3y**2)
        b_psi = P.ellT * f_e / (J_T + P.J1z)

        M = 6
        tr_yaw = M * tr_roll
        zeta_yaw = 0.707
        wn_yaw = 0.5 * np.pi / tr_yaw / np.sqrt(1 - zeta_yaw**2)
        self.kp_yaw = wn_yaw**2 / b_psi
        self.kd_yaw = 2 * zeta_yaw * wn_yaw / b_psi

        print('kp_yaw: ', self.kp_yaw)
        print('kd_yaw: ', self.kd_yaw) 

    def update(self, r: np.ndarray, y: np.ndarray):
        theta_ref = r[0][0]
        psi_ref = r[1][0]

        phi = y[0][0]
        theta = y[1][0]
        psi = y[2][0]
        phidot = y[3][0]
        thetadot = y[4][0]
        psidot = y[5][0]

        # lognitudinal control
        # compute errors
        error_theta = theta_ref - theta
        # update differentiators
        self.theta_dot = (theta - self.theta_d1) / self.Ts
        
        # update integrators
        # self.integrator_theta = 

        force_unsat = self.kp_pitch * error_theta - self.kd_pitch * self.theta_dot
        force = saturate(force_unsat, -P.force_max, P.force_max)

        # lateral control
        # outer loop / yaw
        error_psi = psi_ref - psi
        phi_ref = self.kp_yaw * error_psi - self.kd_yaw * psidot
        # inner loop / roll
        error_phi = phi_ref - phi
        torque_unsat = self.kp_roll * error_phi - self.kd_roll * phidot
        torque = saturate(torque_unsat, -P.torque_max, P.torque_max)

        # convert force and torque to pwm signals
        pwm = np.array([[force + torque / P.d],               # u_left
                      [force - torque / P.d]]) / (2 * P.km)   # r_right          
        pwm = saturate(pwm, 0, 1)
        # update all delayed variables
        self.theta_d1 = theta
        self.error_theta_d1 = error_theta
        # return pwm plus reference signals
        return pwm, np.array([[0.], [theta_ref], [psi_ref]])


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




