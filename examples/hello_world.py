import numpy as np
import matplotlib.pyplot as plt
from pybezier import BezierCurve

# generate random curve
np.random.seed(0)
points = np.random.rand(5, 2)
initial_time = 2
final_time = 5
curve = BezierCurve(points, initial_time, final_time)

# plot curve with initial and final points
plt.figure()
curve.plot_trace_2d()
times = [initial_time, final_time]
for time in times:
    value = curve(time)
    plt.scatter(*value, c='r')
    plt.text(*value, f"time = {time}")
plt.show()
