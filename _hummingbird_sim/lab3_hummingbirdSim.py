import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from hummingbirdDynamics import HummingbirdDynamics
from ctrlEquilibrium import ctrlEquilibrium

# instantiate reference input classes
phi_ref = SignalGenerator(amplitude=1.5, frequency=0.05)
theta_ref = SignalGenerator(amplitude=0.5, frequency=0.05)
psi_ref = SignalGenerator(amplitude=0.5, frequency=.05)

# instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()
dynamics = HummingbirdDynamics()
controller = ctrlEquilibrium()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop

    u = controller.update(dynamics.state)

    # update animation
    state = dynamics.update(u)
    ref = np.array([[0], [0], [0]])
    force = 0
    torque = 0
    # convert force and torque to pwm values
    # pwm = P.mixing @ np.array([[force], [torque]]) / P.km
    pwm = u
    animation.update(t, state)
    dataPlot.update(t, dynamics.state, pwm)

    t += P.t_plot  # advance time by t_plot
    plt.pause(0.05)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
