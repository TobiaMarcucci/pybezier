import numpy as np
import matplotlib.pyplot as plt
from pybezier import BezierCurve, CompositeBezierCurve

# generate random continuous composite curve
np.random.seed(0)
n_points = 3
n_curves = 5
dimension = 2
points = np.random.rand((n_points - 1) * n_curves + 1, dimension)
curves = []
for i in range(n_curves):
    start = (n_points - 1) * i
    stop = start + n_points
    curves.append(BezierCurve(points[start:stop], i, i + 1))
composite_curve = CompositeBezierCurve(curves)

# plot curve trace and control points
plt.figure()
composite_curve.plot_trace_2d()
composite_curve.scatter_points_2d()
plt.show()
