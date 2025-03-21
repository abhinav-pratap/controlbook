import numpy as np
import hummingbirdParam as P

class ctrlEquilibrium:
    def __init__(self):
        pass 

    def update(self, x):
        theta = x[1][0]
        thetadot = x[4][0]
        
        force_equilibrium = (P.m1*P.ell1 + P.m2*P.ell2)*P.g/P.ellT
        force = force_equilibrium
        torque = 0.
        # convert force and torque to pwm signals
        pwm = np.array([
            [(0.5 * (1/P.km)) * (force + torque/P.d)],
            [(0.5 * (1/P.km)) * (force - torque/P.d)]
        ])
        pwm = saturate(pwm, 0, 1) 
        return pwm


def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        u = np.max((np.min((u, up_limit)), low_limit))
    else:
        for i in range(0, u.shape[0]):
            u[i][0] = np.max((np.min((u[i][0], up_limit)), low_limit))
    return u




