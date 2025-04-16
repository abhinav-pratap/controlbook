import numpy as np 
import blockbeamParam as P


class blockbeamDynamics:
    def __init__(self, alpha=0.0):
        # Initial state conditions
        self.state = np.array([
            [P.z0],
            [P.theta0],
            [P.zdot0],
            [P.thetadot0],
        ])

        # vary the actual physical parameters
        self.m1 = P.m1 * (1.+alpha*(2.*np.random.rand()-1.))
        self.m2 = P.m2 * (1.+alpha*(2.*np.random.rand()-1.))
        self.length = P.length * (1.+alpha*(2.*np.random.rand()-1.))

        # P.m1 = self.m1
        # P.m2 = self.m2
        # P.length = self.length
        
 
    def update(self, u: float):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        # saturate the input force
        u = saturate(u, P.F_max)
        self.rk4_step(u)  # propagate the state by one time sample
        y = self.h()  # return the corresponding output
        return y

    def f(self, state: np.ndarray, u: float):
        # Return xdot = f(x,u)
        zdot = state[2][0]
        thetadot = state[3][0]

        z = state[0][0]
        theta = state[1][0]

        zddot = z * thetadot**2 - P.g * np.sin(theta)
        thetaddot = (u*self.length*np.cos(theta) - self.m1 * P.g * z * np.cos(theta) - self.length * 0.5 * self.m2 * P.g * np.cos(theta) - 2 * self.m1 * z * zdot * thetadot) / (self.m1 * z**2 + (1/3) * self.m2 * self.length**2)
        
        # build xdot and return
        xdot = np.array([[zdot],
                         [thetadot],
                         [zddot],
                         [thetaddot],
                        ])
        return xdot

    def h(self):
        # FIXME Fill in this function
        # return y = h(x)
        z = self.state[0][0]
        theta = self.state[1][0]
        y = np.array([[z], [theta]])
        return y

    def rk4_step(self, u: float):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + P.Ts / 2 * F1, u)
        F3 = self.f(self.state + P.Ts / 2 * F2, u)
        F4 = self.f(self.state + P.Ts * F3, u)
        self.state = self.state + P.Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)


def saturate(u: float, limit: float):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u
