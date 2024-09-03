import numpy as np
from typing import List, Self
from collections.abc import Iterable
from numbers import Number
from pybezier.bezier_curve import BezierCurve

class CompositeBezierCurve(object):

    def __init__(self, curves):
        initial_times = [curve.initial_time for curve in curves[1:]]
        final_times = [curve.final_time for curve in curves[:-1]]
        if not np.allclose(initial_times, final_times):
            raise ValueError("Initial and final times don't match.")
        dimensions = [curve.dimension for curve in curves]
        if len(set(dimensions)) != 1:
            raise ValueError("All the curves must have the same dimension.")
        self.curves = curves
        self.dimension = curves[0].dimension
        self.initial_time = curves[0].initial_time
        self.final_time = curves[-1].final_time
        self.duration = self.final_time - self.initial_time
        self.knot_times = [self.initial_time] + [curve.final_time for curve in curves]

    def initial_point(self) -> np.array:
        return self[0].initial_point()

    def final_point(self) -> np.array:
        return self[-1].final_point()

    def __iter__(self) -> Iterable[BezierCurve]:
        return iter(self.curves)

    def __getitem__(self, i : int) -> BezierCurve:
        return self.curves[i]

    def __call__(self, time : float) -> np.array:
        segment = self.find_segment(time)
        return self[segment](time)
    
    def find_segment(self, time : float) -> int:
        segment = 0
        while self[segment].final_time < time:
            segment += 1
        return segment

    def __len__(self) -> int:
        return(len(self.curves))
    
    def __mul__(self, composite_curve : Self | Number) -> Self:
        if not isinstance(composite_curve, CompositeBezierCurve):
            composite_curve = self._number_to_composite_curve(composite_curve)
        curves = [curve_1 * curve_2 for curve_1, curve_2 in zip(self, composite_curve)]
        return CompositeBezierCurve(curves)

    def _number_to_composite_curve(self, n : Number) -> Self:
        curves = [curve._number_to_curve(n) for curve in self]
        return CompositeBezierCurve(curves)

    def __rmul__(self, composite_curve : Self | Number) -> Self:
        return self * composite_curve
    
    def elevate_degree(self, degree : int) -> Self:
        curves = [curve.elevate_degree(degree) for curve in self]
        return CompositeBezierCurve(curves)

    def __add__(self, composite_curve : Self | Number) -> Self:
        if not isinstance(composite_curve, CompositeBezierCurve):
            composite_curve = self._number_to_composite_curve(composite_curve)
        curves = [curve_1 + curve_2 for curve_1, curve_2 in zip(self, composite_curve)]
        return CompositeBezierCurve(curves)
    
    def __radd__(self, composite_curve : Self | Number) -> Self:
        return self + composite_curve
    
    def __sub__(self, composite_curve : Self | Number) -> Self:
        return self + composite_curve * (-1)
    
    def __rsub__(self, composite_curve : Self | Number) -> Self:
        return self * (-1) + composite_curve
    
    def  __neg__(self) -> Self:
        return 0 - self

    def knot_points(self) -> np.array:
        knots = [curve.points[0] for curve in self]
        knots.append(self.final_point())
        return np.array(knots)

    def durations(self) -> np.array:
        return np.array([curve.duration for curve in self])

    def concatenate(self, composite_curve : Self) -> Self:
        shifted_curves = []
        for curve in composite_curve:
            initial_time = curve.initial_time + self.duration
            final_time = curve.final_time + self.duration
            shifted_curve = BezierCurve(curve.points, initial_time, final_time)
            shifted_curves.append(shifted_curve)
        return CompositeBezierCurve(self.curves + shifted_curves)

    def derivative(self) -> Self:
        return CompositeBezierCurve([curve.derivative() for curve in self])
    
    def l2_squared(self) -> float:
        return sum(curve.l2_squared() for curve in self)

    def plot_components(self, n : int = 51, legend : bool = True, **kwargs):
        for i, curve in enumerate(self):
            curve.plot_components(n, legend, **kwargs)
            if i == 0:
                legend = False
        
    def plot_trace_2d(self, **kwargs):
        for curve in self:
            curve.plot_trace_2d(**kwargs)

    def scatter_points_2d(self, **kwargs):
        for curve in self:
            curve.scatter_points_2d(**kwargs)

    def plot_control_polytopes_2d(self, **kwargs):
        for curve in self:
            curve.plot_control_polytope_2d(**kwargs)
