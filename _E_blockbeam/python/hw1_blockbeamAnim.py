from blockbeamAnimation import blockbeamAnimation
import blockbeamParam as P
import matplotlib.pyplot as plt
import numpy as np
from signalGenerator import signalGenerator
from dataPlotter import dataPlotter


z_plot = signalGenerator(amplitude=P.length/4., frequency=0.5, y_offset=P.length/2.)
theta_plot = signalGenerator(amplitude=np.pi/8, frequency=0.1, y_offset=0.0)
f_plot = signalGenerator(amplitude=5, frequency=.5)

data_plot = dataPlotter()
animation = blockbeamAnimation()
t = P.t_start

while t < P.t_end:
    z = z_plot.sin(t)
    theta = theta_plot.sin(t)
    f = f_plot.sawtooth(t)

    state = np.array([[z], [theta], [0.0], [0.0]])
    animation.update(state)
    data_plot.update(t, state, f)

    t += P.t_plot  # advance time by t_plot
    plt.pause(0.05)

plt.waitforbuttonpress()
plt.close()