import numpy as np
from typing import Tuple, List, Callable, Self
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
        self.transition_times = [self.initial_time] + [curve.final_time for curve in curves]

    def curve_segment(self, time : float) -> int:
        segment = 0
        while self[segment].final_time < time:
            segment += 1
        return segment

    def __call__(self, time : float) -> np.ndarray:
        segment = self.curve_segment(time)
        return self[segment](time)

    @property
    def initial_point(self) -> np.ndarray:
        return self[0].initial_point

    @property
    def final_point(self) -> np.ndarray:
        return self[-1].final_point

    def __iter__(self) -> Iterable[BezierCurve]:
        return iter(self.curves)

    def __getitem__(self, i : int) -> BezierCurve:
        return self.curves[i]

    def __len__(self) -> int:
        return(len(self.curves))
    
    def __mul__(self, composite_curve : Self | Number) -> Self:
        if isinstance(composite_curve, Number):
            composite_curve = self._number_to_composite_curve(composite_curve)
        curves = [curve_1 * curve_2 for curve_1, curve_2 in zip(self, composite_curve)]
        return CompositeBezierCurve(curves)

    def _number_to_composite_curve(self, n : Number) -> Self:
        curves = [curve._number_to_curve(n) for curve in self]
        return CompositeBezierCurve(curves)

    def _array_to_composite_curve(self, x : np.ndarray) -> Self:
        curves = [curve._array_to_curve(x) for curve in self]
        return CompositeBezierCurve(curves)

    def __rmul__(self, composite_curve : Self | Number) -> Self:
        return self * composite_curve
    
    def elevate_degree(self, degree : int) -> Self:
        curves = [curve.elevate_degree(degree) for curve in self]
        return CompositeBezierCurve(curves)

    def __add__(self, composite_curve : Self | Number) -> Self:
        if isinstance(composite_curve, Number):
            composite_curve = self._number_to_composite_curve(composite_curve)
        if isinstance(composite_curve, np.ndarray):
            composite_curve = self._array_to_composite_curve(composite_curve)
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

    def transition_points(self) -> np.ndarray:
        # assumes that the curve is continuous
        points = [curve.initial_point for curve in self]
        points.append(self.final_point)
        return np.array(points)

    def durations(self) -> np.ndarray:
        return np.array([curve.duration for curve in self])

    def concatenate(self, composite_curve : Self) -> Self:
        t = self.final_time - composite_curve.initial_time
        shifted_curves = composite_curve.time_shift(t).curves
        return CompositeBezierCurve(self.curves + shifted_curves)

    def domain_split(self, time : float) -> Tuple[Self, Self]:
        if time < self.initial_time:
            raise ValueError("Split time must be greater than or equal to initial time.")
        elif time == self.initial_time:
            return None, self
        if time > self.final_time:
            raise ValueError("Split time must be lower than or equal to final time.")
        elif time == self.final_time:
            return self, None
        segment = self.curve_segment(time)
        curves1 = self[:segment]
        curves2 = self[segment+1:]
        curve1, curve2 = self[segment].domain_split(time)
        if curve1 is not None:
            curves1.append(curve1)
        if curve2 is not None:
            curves2.insert(0, curve2)
        return CompositeBezierCurve(curves1), CompositeBezierCurve(curves2)

    def time_shift(self, t : float) -> Self:
        curves = []
        for curve in self:
            curves.append(curve.time_shift(t))
        return CompositeBezierCurve(curves)

    def derivative(self) -> Self:
        return CompositeBezierCurve([curve.derivative() for curve in self])

    def integral(self, initial_condition : np.ndarray | None = None) -> Self:
        curves = []
        for curve in self:
            curves.append(curve.integral(initial_condition))
            initial_condition = curves[-1].final_point
        return CompositeBezierCurve(curves)
    
    def l2_squared(self) -> float:
        return sum(curve.l2_squared() for curve in self)

    def integral_of_convex_function(self, f : Callable) -> float:
        return sum(curve.integral_of_convex_function(f) for curve in self)

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
