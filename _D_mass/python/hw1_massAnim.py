from massAnimation import massAnimation
import massParam as P
import matplotlib.pyplot as plt
from signalGenerator import signalGenerator

animation = massAnimation()

z_plot = signalGenerator(amplitude=3, frequency=0.1)

t = P.t_start

while t < P.t_end:
    z = z_plot.sin(t)
    animation.update([[z]])
    t += P.t_plot  # advance time by t_plot
    plt.pause(0.05)

plt.waitforbuttonpress()
plt.close()