from blockbeamAnimation import blockbeamAnimation
import blockbeamParam as P
import matplotlib.pyplot as plt
import numpy as np
from signalGenerator import signalGenerator
from dataPlotter import dataPlotter
from blockbeamDynamics import blockbeamDynamics
from ctrlPd import ctrlPD

r_plot = signalGenerator(amplitude=0.15, frequency=.05, y_offset=0.25)
dynamics = blockbeamDynamics()

data_plot = dataPlotter()
animation = blockbeamAnimation()
controller = ctrlPD()
t = P.t_start

while t < P.t_end:
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        r = r_plot.square(t)
        x = dynamics.state
        u = controller.update(r, x)
        state = dynamics.update(u)
        t += P.Ts

    animation.update(dynamics.state)
    data_plot.update(t, dynamics.state, u, r)

    plt.pause(0.0001)

plt.waitforbuttonpress()
plt.close()