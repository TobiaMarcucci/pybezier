import numpy as np
import operator
from typing import List, Callable, Union
from numbers import Number
from pybezier.binomial import binomial

class BezierCurve(object):

    def __init__(self, points : np.ndarray, initial_time : float = 0, final_time : float = 1):
        if len(points.shape) < 2:
            raise ValueError("Array of control points must be at least two dimensional.")
        if initial_time >= final_time:
            raise ValueError("Initial time must be smaller than final time.")
        self.points = points
        self.initial_time = initial_time
        self.final_time = final_time

    @property
    def degree(self) -> int:
        return len(self.points) - 1

    @property
    def shape(self) -> tuple[int, ...]:
        return self.points[0].shape

    @property
    def size(self) -> int:
        return self.points[0].size

    @property
    def duration(self) -> float:
        return self.final_time - self.initial_time

    @property
    def initial_point(self) -> np.ndarray:
        return self.points[0]

    @property
    def final_point(self) -> np.ndarray:
        return self.points[-1]

    def _berstein(self, time : Union[float, np.ndarray], n : int) -> float:
        c = binomial(self.degree, n)
        t = (time - self.initial_time) / self.duration 
        value = c * t ** n * (1 - t) ** (self.degree - n)
        return value
    
    def _assert_same_times(self, curve : "BezierCurve"):
        if not np.isclose(self.initial_time, curve.initial_time):
            raise ValueError("Incompatible curves, initial times do not match.")
        if not np.isclose(self.final_time, curve.final_time):
            raise ValueError("Incompatible curves, final times do not match.")

    def __call__(self, time : Union[float, np.ndarray]) -> np.ndarray:
        c = np.array([self._berstein(time, n) for n in range(self.degree + 1)])
        return (self.points.T @ c).T

    def _number_to_curve(self, n : Number) -> "BezierCurve":
        points = np.array([[n]])
        return BezierCurve(points, self.initial_time, self.final_time)

    def _array_to_curve(self, x : np.ndarray) -> "BezierCurve":
        points = np.array([x])
        return BezierCurve(points, self.initial_time, self.final_time)

    def _multiplication(self, curve : Union["BezierCurve", Number], mul_op : Callable) -> "BezierCurve":
        """See (44) in Algorithms for polynomials in Bernstein form, by Farouky
        and Rajan"""
        if isinstance(curve, Number):
            curve = self._number_to_curve(curve)
        if isinstance(curve, np.ndarray):
            curve = self._array_to_curve(curve)
        self._assert_same_times(curve)
        degree = self.degree + curve.degree
        # select shape and dtype automatically
        example_product = mul_op(self.points[0], curve.points[0])
        shape = example_product.shape
        dtype = example_product.dtype
        points = np.zeros((degree + 1, *shape), dtype=dtype)
        for i in range(degree + 1):
            j_min = max(0, i - curve.degree)
            j_max = min(self.degree, i)
            for j in range(j_min, j_max + 1):
                b = binomial(self.degree, j)
                b *= binomial(curve.degree, i - j)
                points[i] += mul_op(self.points[j], curve.points[i - j]) * b
            points[i] /= binomial(degree, i)
        return BezierCurve(points, self.initial_time, self.final_time)

    def __mul__(self, curve : Union["BezierCurve", Number, np.ndarray]) -> "BezierCurve":
        return self._multiplication(curve, operator.mul)

    def __rmul__(self, curve : Union["BezierCurve", Number, np.ndarray]) -> "BezierCurve":
        return self * curve

    def __imul__(self, curve : Union["BezierCurve", Number, np.ndarray]) -> "BezierCurve":
        self.points = (self * curve).points
        return self

    def __matmul__(self, curve : Union["BezierCurve", Number, np.ndarray]) -> "BezierCurve":
        return self._multiplication(curve, operator.matmul)
    
    def __add__(self, curve : Union["BezierCurve", Number, np.ndarray]) -> "BezierCurve":
        if isinstance(curve, Number):
            curve = self._number_to_curve(curve)
        if isinstance(curve, np.ndarray):
            curve = self._array_to_curve(curve)
        self._assert_same_times(curve)
        if curve.degree > self.degree:
            self = self.elevate_degree(curve.degree)
        elif self.degree > curve.degree:
            curve = curve.elevate_degree(self.degree)
        points = self.points + curve.points
        return BezierCurve(points, self.initial_time, self.final_time)
    
    def __radd__(self, curve : Union["BezierCurve", Number, np.ndarray]) -> "BezierCurve":
        return self + curve

    def __iadd__(self, curve : Union["BezierCurve", Number, np.ndarray]) -> "BezierCurve":
        self.points = (self + curve).points
        return self
    
    def __sub__(self, curve : Union["BezierCurve", Number, np.ndarray]) -> "BezierCurve":
        return self + curve * (-1)
    
    def __rsub__(self, curve : Union["BezierCurve", Number, np.ndarray]) -> "BezierCurve":
        return self * (-1) + curve

    def __isub__(self, curve : Union["BezierCurve", Number, np.ndarray]) -> "BezierCurve":
        self.points = (self - curve).points
        return self
    
    def  __neg__(self) -> "BezierCurve":
        return 0 - self

    def elevate_degree(self, degree : int) -> "BezierCurve":
        points = np.ones((degree - self.degree + 1, 1))
        curve = BezierCurve(points, self.initial_time, self.final_time)
        return self * curve
    
    def derivative(self) -> "BezierCurve":
        points = (self.points[1:] - self.points[:-1]) * (self.degree / self.duration)
        return BezierCurve(points, self.initial_time, self.final_time)

    def integral(self, initial_condition : np.ndarray | None = None) -> "BezierCurve":
        points = self.points * self.duration / (self.degree + 1)
        points = np.vstack([np.zeros(self.shape), points])
        points = np.cumsum(points, axis=0)
        if initial_condition is not None:
            points += initial_condition
        return BezierCurve(points, self.initial_time, self.final_time)
    
    def domain_split(self, time : float) -> tuple["BezierCurve", "BezierCurve"]:
        if time < self.initial_time:
            raise ValueError("Split time must be greater than or equal to initial time.")
        elif time == self.initial_time:
            return None, self
        elif time > self.final_time:
            raise ValueError("Split time must be lower than or equal to final time.")
        elif time == self.final_time:
            return self, None
        points = self.points
        points1 = np.zeros(self.points.shape, dtype=self.points.dtype)
        points2 = np.zeros(self.points.shape, dtype=self.points.dtype)
        c = (time - self.initial_time) / self.duration
        d = (self.final_time - time) / self.duration
        for i in range(self.degree):
            points1[i] = points[0]
            points2[-i-1] = points[-1]
            points = points[1:] * c + points[:-1] * d
        points1[-1] = points
        points2[0] = points
        curve1 = BezierCurve(points1, self.initial_time, time)
        curve2 = BezierCurve(points2, time, self.final_time)
        return curve1, curve2

    def time_shift(self, t : float) -> "BezierCurve":
        initial_time = self.initial_time + t
        final_time = self.final_time + t
        return BezierCurve(self.points, initial_time, final_time)
    
    def l2_squared(self) -> float:
        """See (34) in Algorithms for polynomials in Bernstein form, by Farouky
        and Rajan"""
        a = 0
        for i in range(self.degree + 1):
            bi = binomial(self.degree, i)
            for j in range(i, self.degree + 1):
                bj = binomial(self.degree, j)
                bij = binomial(2 * self.degree, i + j)
                b = bi * bj / bij
                if j > i:
                    b *= 2
                a += b * self.points[i].dot(self.points[j])
        return self.duration * a / (2 * self.degree + 1)
    
    def integral_of_convex_function(self, f : Callable) -> float:
        c = self.duration / (self.degree + 1)
        return c * sum(f(point) for point in self.points)

    def plot_components(self, n : int = 51, legend : bool = True, **kwargs):
        import matplotlib.pyplot as plt
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        repeat = np.ceil(self.shape / len(colors))
        colors = colors * int(repeat)
        t = np.linspace(self.initial_time, self.final_time, n)
        values = self(t)
        for i, value in enumerate(values.T):
            label = f"component {i}" if legend else None
            plt.plot(t, value, c=colors[i], label=label, **kwargs)
        if legend:
            plt.legend()

    def plot_trace_2d(self, n : int = 51, **kwargs):
        if self.shape != (2,):
            raise ValueError("Can only plot trace of 2d curves.")
        import matplotlib.pyplot as plt
        options = {"c": "b"}
        options.update(kwargs)
        t = np.linspace(self.initial_time, self.final_time, n)
        plt.plot(*self(t).T, **options)

    def scatter_points_2d(self, **kwargs):
        if self.shape != (2,):
            raise ValueError("Can only scatter points of 2d curves.")
        import matplotlib.pyplot as plt
        options = {"fc": "orange", "ec": "k", "zorder": 3}
        options.update(kwargs)
        plt.scatter(*self.points.T, **options)

    def plot_control_polytope_2d(self, **kwargs):
        if self.shape != (2,):
            raise ValueError("Can plot control polytope of 2d curves.")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull
        options = {"fc": "lightcoral"}
        options.update(kwargs)
        hull = ConvexHull(self.points)
        ordered_points = hull.points[hull.vertices]
        poly = Polygon(ordered_points, **options)
        plt.gca().add_patch(poly)
