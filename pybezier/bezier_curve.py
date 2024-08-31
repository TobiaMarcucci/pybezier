import numpy as np
from pybezier.binomial import binomial

class BezierCurve(object):

    def __init__(self, points, a=0, b=1):
        assert b > a
        self.points = points
        self.degree, self.dimension = points.shape
        self.degree -= 1
        self.a = a
        self.b = b
        self.duration = b - a

    def _berstein(self, t, n):
        c1 = binomial(self.degree, n)
        c2 = (t - self.a) / self.duration 
        c3 = (self.b - t) / self.duration
        value = c1 * c2 ** n * c3 ** (self.degree - n)
        return value
    
    def _assert_same_times(self, curve):
        assert np.isclose(self.a, curve.a)
        assert np.isclose(self.b, curve.b)

    def _convert_to_curve(self, curve):
        if not isinstance(curve, BezierCurve):
            points = np.array([[curve]])
            curve = BezierCurve(points, self.a, self.b)
        return curve

    def __call__(self, t):
        c = np.array([self._berstein(t, n) for n in range(self.degree + 1)])
        return c.T.dot(self.points)
    
    def __mul__(self, curve):
        """See (44) in 'Algorithms for polynomials in Bernstein form' by Farouky and Rajan"""
        curve = self._convert_to_curve(curve)
        self._assert_same_times(curve)
        degree = self.degree + curve.degree
        dimension = max(self.dimension, curve.dimension)
        points = np.zeros((degree + 1, dimension), dtype=object)
        for i in range(degree + 1):
            j_min = max(0, i - curve.degree)
            j_max = min(self.degree, i)
            for j in range(j_min, j_max + 1):
                b = binomial(self.degree, j)
                b *= binomial(curve.degree, i - j)
                points[i] += self.points[j] * curve.points[i - j] * b
            points[i] /= binomial(degree, i)
        return BezierCurve(points, self.a, self.b)
    
    def __rmul__(self, curve):
        return self * curve

    def elevate_degree(self, degree):
        points = np.ones((degree - self.degree + 1, 1))
        curve = BezierCurve(points, self.a, self.b)
        return self * curve
    
    def __add__(self, curve):
        curve = self._convert_to_curve(curve)
        self._assert_same_times(curve)
        if curve.degree > self.degree:
            self = self.elevate_degree(curve.degree)
        elif self.degree > curve.degree:
            curve = curve.elevate_degree(self.degree)
        points = self.points + curve.points
        return BezierCurve(points, self.a, self.b)
    
    def __radd__(self, curve):
        return self + curve
    
    def __sub__(self, curve):
        return self + curve * (-1)
    
    def __rsub__(self, curve):
        return self * (-1) + curve
    
    def  __neg__(self):
        return 0 - self

    def start_point(self):
        return self.points[0]

    def end_point(self):
        return self.points[-1]
        
    def derivative(self):
        points = (self.points[1:] - self.points[:-1]) * (self.degree / self.duration)
        return BezierCurve(points, self.a, self.b)

    def split(self, t):
        assert t >= self.a
        assert t <= self.b
        points = self.points
        points1 = np.zeros(self.points.shape)
        points2 = np.zeros(self.points.shape)
        c = (t - self.a) / self.duration
        d = (self.b - t) / self.duration
        for i in range(self.degree):
            points1[i] = points[0]
            points2[-i-1] = points[-1]
            points = points[1:] * c + points[:-1] * d
        points1[-1] = points
        points2[0] = points
        curve1 = BezierCurve(points1, self.a, t)
        curve2 = BezierCurve(points2, t, self.b)
        return curve1, curve2
    
    def l2_squared(self):
        """See (44) in 'Algorithms for polynomials in Bernstein form' by Farouky and Rajan"""
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

    def plot_components(self, n=51, legend=True, **kwargs):
        import matplotlib.pyplot as plt
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        repeat = np.ceil(self.dimension / len(colors))
        colors = colors * int(repeat)
        t = np.linspace(self.a, self.b, n)
        values = self(t)
        for i, value in enumerate(values.T):
            label = f'component {i}' if legend else None
            plt.plot(t, value, c=colors[i], label=label, **kwargs)
        if legend:
            plt.legend()

    def plot_trace_2d(self, n=51, **kwargs):
        assert self.dimension == 2
        import matplotlib.pyplot as plt
        options = {'c':'b'}
        options.update(kwargs)
        t = np.linspace(self.a, self.b, n)
        plt.plot(*self(t).T, **options)

    def scatter_2d(self, **kwargs):
        assert self.dimension == 2
        import matplotlib.pyplot as plt
        options = {'fc':'orange', 'ec':'k', 'zorder':3}
        options.update(kwargs)
        plt.scatter(*self.points.T, **options)

    def plot_control_polytope_2d(self, **kwargs):
        assert self.dimension == 2
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull
        options = {'fc':'lightcoral'}
        options.update(kwargs)
        hull = ConvexHull(self.points)
        ordered_points = hull.points[hull.vertices]
        poly = Polygon(ordered_points, **options)
        plt.gca().add_patch(poly)
