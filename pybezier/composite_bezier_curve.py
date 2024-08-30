import numpy as np
from pybezier.bezier_curve import BezierCurve

class CompositeBezierCurve:

    def __init__(self, beziers):
        for curve1, curve2 in zip(beziers[:-1], beziers[1:]):
            assert np.isclose(curve1.b, curve2.a)
            assert curve1.dimension == curve2.dimension
        self.beziers = beziers
        self.N = len(self.beziers)
        self.dimension = beziers[0].dimension
        self.a = beziers[0].a
        self.b = beziers[-1].b
        self.duration = self.b - self.a
        self.transition_times = [self.a] + [bez.b for bez in beziers]

    def __iter__(self):
        return iter(self.beziers)

    def __getitem__(self, i):
        return self.beziers[i]

    def __call__(self, t):
        i = self.find_segment(t)
        return self[i](t)

    def __len__(self):
        return(len(self.beziers))

    def start_point(self):
        return self[0].start_point()

    def end_point(self):
        return self[-1].end_point()

    def knot_points(self):
        knots = [bez.points[0] for bez in self]
        return np.array(knots + [self[-1].points[-1]])

    def durations(self):
        return np.array([bez.duration for bez in self])

    def concatenate(self, curve):
        shifted_beziers = [BezierCurve(b.points, b.a + self.duration, b.b + self.duration) for b in curve]
        return CompositeBezierCurve(self.beziers + shifted_beziers)

    def derivative(self):
        return CompositeBezierCurve([b.derivative() for b in self])
    
    def l2_squared(self):
        return sum(bez.l2_squared() for bez in self)

    def plot_components(self, samples=51, **kwargs):
        for i, bez in enumerate(self):
            legend = True if i ==0 else False
            bez.plot_components(samples, legend, **kwargs)
        
    def plot2d(self, **kwargs):
        for bez in self:
            bez.plot2d(**kwargs)

    def scatter2d(self, **kwargs):
        for bez in self:
            bez.scatter2d(**kwargs)

    def plot_2dpolygons(self, **kwargs):
        for bez in self:
            bez.plot_2dpolygon(**kwargs)
