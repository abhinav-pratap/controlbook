from blockbeamAnimation import blockbeamAnimation
import blockbeamParam as P
import matplotlib.pyplot as plt
import numpy as np
from signalGenerator import signalGenerator
from dataPlotter import dataPlotter
from blockbeamDynamics import blockbeamDynamics
from ctrlPd import ctrlPD

r_plot = signalGenerator(amplitude=0.15, frequency=.01, y_offset=0.25)
dynamics = blockbeamDynamics()

data_plot = dataPlotter()
animation = blockbeamAnimation()
controller = ctrlPD()
t = P.t_start

while t < P.t_end:
    r = r_plot.square(t)
    x = dynamics.state
    u = controller.update(r, x)
    state = dynamics.update(u)

    animation.update(state)
    data_plot.update(t, dynamics.state, u, r)

    t += P.t_plot  # advance time by t_plot
    plt.pause(0.05)

plt.waitforbuttonpress()
plt.close()