from blockbeamAnimation import blockbeamAnimation
import blockbeamParam as P
import matplotlib.pyplot as plt
import numpy as np
from signalGenerator import signalGenerator


theta_plot = signalGenerator(amplitude=np.pi/2, frequency=0.01)
z_plot = signalGenerator(amplitude=.25, frequency=.1, y_offset=0.25)


animation = blockbeamAnimation()
t = P.t_start

while t < P.t_end:
    z = z_plot.sin(t)
    theta = theta_plot.sin(t)
    animation.update([[z], [theta]])
    t += P.t_plot  # advance time by t_plot
    plt.pause(0.05)

plt.waitforbuttonpress()
plt.close()