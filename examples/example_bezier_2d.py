import numpy as np
import matplotlib.pyplot as plt
from pybezier import BezierCurve

# generate random curve
np.random.seed(0)
control_points = np.random.rand(5, 2) # 5 points in 2d
initial_time = 2
final_time = 5
curve = BezierCurve(control_points, initial_time, final_time)

# plot curve trace and control points
plt.figure()
curve.plot_trace_2d()
curve.scatter_points_2d()
plt.show()

# print a variety of properties
print('Curve degree:', curve.degree)
print('Curve shape:', curve.shape)
print('Curve dimension:', curve.dimension)
print('Curve initial time:', curve.initial_time)
print('Curve final time:', curve.final_time)
print('Curve duration:', curve.duration)
print('Curve control points:', curve.points)
print('Curve initial point:', curve.initial_point)
print('Curve final point:', curve.final_point)

# operations on and between curves
curve_sum = curve + curve
curve_sub = curve - curve
curve_neg = - curve
curve_prod = curve * 3
curve_elementwise_prod = curve * curve
curve_scalar_prod = curve @ curve
curve_elevate_degree = curve.elevate_degree(curve.degree * 2)
derivative = curve.derivative
integral = curve.integral(initial_condition=np.ones(curve.dimension))
curve_1, curve_2 = curve.split_domain((initial_time + final_time) / 2)
shifted_curve = curve.shift_domain(.5)
squared_l2_norm = curve.squared_l2_norm()
