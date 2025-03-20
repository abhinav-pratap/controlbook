import numpy as np
from massAnimation import massAnimation
import massParam as P
import matplotlib.pyplot as plt
from signalGenerator import signalGenerator
from dataPlotter import dataPlotter
from massDynamics import massDynamics
from ctrlPd import ctrlPD

dynamics = massDynamics()
animation = massAnimation()
data_plot = dataPlotter()
ctrl = ctrlPD()

r_plot = signalGenerator(amplitude=1, frequency=0.05)

t = P.t_start

while t < P.t_end:
    r = r_plot.square(t)
    u = ctrl.update(r, dynamics.state)
    dynamics.update(u)

    animation.update(dynamics.state)
    data_plot.update(t, dynamics.state, u, r)
    
    t += P.t_plot  # advance time by t_plot
    plt.pause(0.05)

plt.waitforbuttonpress()
plt.close()