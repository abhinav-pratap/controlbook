import numpy as np 
import massParam as P


class massDynamics:
    def __init__(self, alpha=0.0):
        # Initial state conditions
        self.state = np.array([
            [P.z0],
            [P.zdot0], 
        ])

        # vary the actual physical parameters
        self.k = P.k * (1.+alpha*(2.*np.random.rand()-1.))
        self.m = P.m * (1.+alpha*(2.*np.random.rand()-1.))
        self.b = P.b * (1.+alpha*(2.*np.random.rand()-1.))
 
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
        z = state[0][0]
        zdot = state[1][0]

        zddot = (u - self.b*zdot - self.k*z) / self.m
        
        # build xdot and return
        xdot = np.array([[zdot],
                         [zddot]
                        ])
        return xdot

    def h(self):
        # FIXME Fill in this function
        # return y = h(x)
        z = self.state[0][0]
        y = np.array([[z]])
        return y

    def rk4_step(self, u: float):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + P.Ts / 2 * F1, u)
        F3 = self.f(self.state + P.Ts / 2 * F2, u)
        F4 = self.f(self.state + P.Ts * F3, u)
        self.state = self.state + P.Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)


def saturate(u: np.ndarray, limit: float):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u
