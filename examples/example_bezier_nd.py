import numpy as np
from pybezier import BezierCurve

# generate random scalar curve
np.random.seed(0)
control_points = np.random.rand(5) # 5 points in 1d
initial_time = 2
final_time = 5
curve = BezierCurve(control_points, initial_time, final_time)

# print value at intermediate time
intermediate_time = (initial_time + final_time) / 2
print('Scalar curve at intermediate time:', curve(intermediate_time))

# generate random matrix-valued curve
control_points = np.random.rand(5, 4, 3) # 5 points in (4x3)d
curve = BezierCurve(control_points, initial_time, final_time)
print('Matrix curve at intermediate time:', curve(intermediate_time))
