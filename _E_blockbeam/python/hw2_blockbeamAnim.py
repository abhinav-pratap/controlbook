from blockbeamAnimation import blockbeamAnimation
import blockbeamParam as P
import matplotlib.pyplot as plt
import numpy as np
from signalGenerator import signalGenerator
from dataPlotter import dataPlotter
from blockbeamDynamics import blockbeamDynamics

f_plot = signalGenerator(amplitude=-0.1, frequency=.2, y_offset=9)
dynamics = blockbeamDynamics()

data_plot = dataPlotter()
animation = blockbeamAnimation()
t = P.t_start

while t < P.t_end:
    u = f_plot.sawtooth(t)
    # u = 9.75
    state = dynamics.update(u)

    animation.update(state)
    data_plot.update(t, dynamics.state, u)

    t += P.t_plot  # advance time by t_plot
    plt.pause(0.05)

plt.waitforbuttonpress()
plt.close()