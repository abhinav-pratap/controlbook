import numpy as np
from massAnimation import massAnimation
import massParam as P
import matplotlib.pyplot as plt
from signalGenerator import signalGenerator
from dataPlotter import dataPlotter

animation = massAnimation()
data_plot = dataPlotter()

z_plot = signalGenerator(amplitude=1, frequency=0.5, y_offset=0.2)
f_plot = signalGenerator(amplitude=2, frequency=0.5)

t = P.t_start

while t < P.t_end:
    z = z_plot.sin(t)
    f = f_plot.sawtooth(t)

    state = np.array([[z], [0.0]])
    animation.update(state)
    data_plot.update(t, state, f)
    
    t += P.t_plot  # advance time by t_plot
    plt.pause(0.05)

plt.waitforbuttonpress()
plt.close()