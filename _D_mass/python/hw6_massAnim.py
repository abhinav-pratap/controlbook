import numpy as np
from massAnimation import massAnimation
import massParam as P
import matplotlib.pyplot as plt
from signalGenerator import signalGenerator
from dataPlotter import dataPlotter
from massDynamics import massDynamics
from ctrlPID import ctrlPID

dynamics = massDynamics(alpha=0.2)
animation = massAnimation()
data_plot = dataPlotter()
ctrl = ctrlPID()

r_plot = signalGenerator(amplitude=1, frequency=0.05)

t = P.t_start

y = dynamics.h()
while t < P.t_end:  # main simulation loop

    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    while t < t_next_plot:  # updates control and dynamics at faster simulation rate
        r = r_plot.square(t)  # reference input
        u = ctrl.update(r, y)  # update controller
        y = dynamics.update(u)  # propagate system
        t += P.Ts  # advance time by Ts
        
    # update animation and data plots
    animation.update(dynamics.state)
    data_plot.update(t, dynamics.state, u, r)
    plt.pause(0.0001)  # the pause causes the figure to be displayed during the simulation

plt.waitforbuttonpress()
plt.close()