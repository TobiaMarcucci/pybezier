import numpy as np
from typing import Tuple, Callable, Union, Optional
from collections.abc import Iterable
from numbers import Number
from pybezier.bezier import BezierCurve

class CompositeBezierCurve(object):

    def __init__(self, curves):
        initial_times = [curve.initial_time for curve in curves[1:]]
        final_times = [curve.final_time for curve in curves[:-1]]
        if not np.allclose(initial_times, final_times):
            raise ValueError("Initial and final times don't match.")
        shapes = [curve.shape for curve in curves]
        if len(set(shapes)) != 1:
            raise ValueError("All the curves must have the same shape.")
        self.curves = curves

    @property
    def degrees(self) -> list[int, ...]:
        return [curve.degree for curve in self]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.curves[0].shape

    @property
    def dimension(self) -> int:
        return self.curves[0].dimension

    @property
    def initial_time(self) -> float:
        return self.curves[0].initial_time

    @property
    def final_time(self) -> float:
        return self.curves[-1].final_time

    @property
    def transition_times(self) -> np.array:
        times = [curve.initial_time for curve in self]
        times.append(self.final_time)
        return np.array(times)

    @property
    def duration(self) -> float:
        return self.final_time - self.initial_time

    @property
    def durations(self) -> np.ndarray:
        return [curve.duration for curve in self]

    @property
    def initial_point(self) -> np.ndarray:
        return self[0].initial_point

    @property
    def final_point(self) -> np.ndarray:
        return self[-1].final_point

    @property
    def transition_points(self) -> np.ndarray:
        # assumes that the curve is continuous
        points = [curve.initial_point for curve in self]
        points.append(self.final_point)
        return np.array(points)

    def segment_index(self, time : float) -> int:
        segment = 0
        while self[segment].final_time < time:
            segment += 1
        return segment

    def __call__(self, time : float) -> np.ndarray:
        segment = self.segment_index(time)
        return self[segment](time)

    def __iter__(self) -> Iterable[BezierCurve]:
        return iter(self.curves)

    def __getitem__(self, i : int) -> BezierCurve:
        return self.curves[i]

    def __len__(self) -> int:
        return(len(self.curves))
    
    def __mul__(self, composite_curve : Union["CompositeBezierCurve", Number, np.ndarray]) -> "CompositeBezierCurve":
        if isinstance(composite_curve, Number) or isinstance(composite_curve, np.ndarray):
            curves = [curve * composite_curve for curve in self]
        else:
            curves = [curve_1 * curve_2 for curve_1, curve_2 in zip(self, composite_curve)]
        return CompositeBezierCurve(curves)

    def __rmul__(self, composite_curve : Union["CompositeBezierCurve", Number, np.ndarray]) -> "CompositeBezierCurve":
        return self * composite_curve

    def __imul__(self, composite_curve : Union["CompositeBezierCurve", Number, np.ndarray]) -> "CompositeBezierCurve":
        if isinstance(composite_curve, Number) or isinstance(composite_curve, np.ndarray):
            for curve in self:
                curve *= composite_curve
        else:
            for curve_1, curve_2 in zip(self, composite_curve):
                curve_1 *= curve_2
        return self

    def __matmul__(self, composite_curve : Union["CompositeBezierCurve", Number, np.ndarray]) -> "CompositeBezierCurve":
        if isinstance(composite_curve, Number) or isinstance(composite_curve, np.ndarray):
            curves = [curve @ composite_curve for curve in self]
        else:
            curves = [curve_1 @ curve_2 for curve_1, curve_2 in zip(self, composite_curve)]
        return CompositeBezierCurve(curves)

    def __add__(self, composite_curve : Union["CompositeBezierCurve", Number, np.ndarray]) -> "CompositeBezierCurve":
        if isinstance(composite_curve, Number) or isinstance(composite_curve, np.ndarray):
            curves = [curve + composite_curve for curve in self]
        else:
            curves = [curve_1 + curve_2 for curve_1, curve_2 in zip(self, composite_curve)]
        return CompositeBezierCurve(curves)
    
    def __radd__(self, composite_curve : Union["CompositeBezierCurve", Number, np.ndarray]) -> "CompositeBezierCurve":
        return self + composite_curve

    def __iadd__(self, composite_curve : Union["CompositeBezierCurve", Number, np.ndarray]) -> "CompositeBezierCurve":
        if isinstance(composite_curve, Number) or isinstance(composite_curve, np.ndarray):
            for curve in self:
                curve += composite_curve
        else:
            for curve_1, curve_2 in zip(self, composite_curve):
                curve_1 += curve_2
        return self
    
    def  __neg__(self) -> "CompositeBezierCurve":
        return self * (-1)
    
    def __sub__(self, composite_curve : Union["CompositeBezierCurve", Number, np.ndarray]) -> "CompositeBezierCurve":
        return self + (- composite_curve)
    
    def __rsub__(self, composite_curve : Union["CompositeBezierCurve", Number, np.ndarray]) -> "CompositeBezierCurve":
        return (- self) + composite_curve

    def __isub__(self, composite_curve : Union["CompositeBezierCurve", Number, np.ndarray]) -> "CompositeBezierCurve":
        self += (- composite_curve)
        return self

    def elevate_degree(self, degree : int) -> "CompositeBezierCurve":
        curves = [curve.elevate_degree(degree) for curve in self]
        return CompositeBezierCurve(curves)

    def concatenate(self, composite_curve : "CompositeBezierCurve") -> "CompositeBezierCurve":
        t = self.final_time - composite_curve.initial_time
        shifted_curves = composite_curve.shift_domain(t).curves
        return CompositeBezierCurve(self.curves + shifted_curves)

    def split_domain(self, time : float) -> Tuple["CompositeBezierCurve", "CompositeBezierCurve"]:
        if time < self.initial_time:
            raise ValueError("Split time must be greater than or equal to initial time.")
        elif time == self.initial_time:
            return None, self
        if time > self.final_time:
            raise ValueError("Split time must be lower than or equal to final time.")
        elif time == self.final_time:
            return self, None
        segment = self.segment_index(time)
        curves1 = self[:segment]
        curves2 = self[segment+1:]
        curve1, curve2 = self[segment].split_domain(time)
        if curve1 is not None:
            curves1.append(curve1)
        if curve2 is not None:
            curves2.insert(0, curve2)
        return CompositeBezierCurve(curves1), CompositeBezierCurve(curves2)

    def shift_domain(self, t : float) -> "CompositeBezierCurve":
        curves = []
        for curve in self:
            curves.append(curve.shift_domain(t))
        return CompositeBezierCurve(curves)

    def derivative(self) -> "CompositeBezierCurve":
        return CompositeBezierCurve([curve.derivative() for curve in self])

    def integral(self, initial_condition : Optional[np.ndarray] = None) -> "CompositeBezierCurve":
        curves = []
        for curve in self:
            curves.append(curve.integral(initial_condition))
            initial_condition = curves[-1].final_point
        return CompositeBezierCurve(curves)
    
    def squared_l2_norm(self) -> float:
        return sum(curve.squared_l2_norm() for curve in self)

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
