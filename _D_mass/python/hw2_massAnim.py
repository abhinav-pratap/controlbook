import numpy as np
from massAnimation import massAnimation
import massParam as P
import matplotlib.pyplot as plt
from signalGenerator import signalGenerator
from dataPlotter import dataPlotter
from massDynamics import massDynamics

dynamics = massDynamics()
animation = massAnimation()
data_plot = dataPlotter()

z_plot = signalGenerator(amplitude=1, frequency=0.5, y_offset=0.2)
f_plot = signalGenerator(amplitude=10, frequency=0.1)

t = P.t_start

while t < P.t_end:
    u = f_plot.sawtooth(t)
    state = dynamics.update(u)

    animation.update(state)
    data_plot.update(t, dynamics.state, u)
    
    t += P.t_plot  # advance time by t_plot
    plt.pause(0.05)

plt.waitforbuttonpress()
plt.close()