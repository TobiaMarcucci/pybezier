import unittest
import numpy as np
from pybezier.bezier_curve import BezierCurve
from pybezier.composite_bezier_curve import CompositeBezierCurve

class TestCompositeBezierCurve(unittest.TestCase):

    @staticmethod
    def random_composite_curve(dimension, n_curves, n_points):
        points = np.random.rand((n_points - 1) * n_curves + 1, dimension)
        curves = []
        for i in range(n_curves):
            start = n_points * i - i
            stop = n_points * (i + 1) - i
            points_i = points[start:stop]
            curve_i = BezierCurve(points_i, i, i + 1)
            curves.append(curve_i)
        return CompositeBezierCurve(curves)

    def setUp(self):
        np.random.seed(0)
        self.dimension = 3
        self.n_curves = 4
        self.n_points = 5
        self.composite_curve = self.random_composite_curve(self.dimension, self.n_curves, self.n_points)
        self.initial_time = 0
        self.final_time = self.n_curves
        self.time_samples = np.linspace(self.initial_time, self.final_time)
        self.composite_curve_1 = self.composite_curve
        self.composite_curve_2 = self.random_composite_curve(self.dimension, self.n_curves, self.n_points)

    def test_init(self):
        self.assertEqual(len(self.composite_curve.curves), self.n_curves)
        for curve in self.composite_curve.curves:
            self.assertEqual(curve.points.shape, (self.n_points, self.dimension))
        self.assertEqual(self.composite_curve.dimension, self.dimension)
        self.assertEqual(self.composite_curve.initial_time, 0)
        self.assertEqual(self.composite_curve.final_time, self.n_curves)
        self.assertEqual(self.composite_curve.duration, self.n_curves)
        self.assertEqual(self.composite_curve.knot_times, list(range(self.n_curves + 1)))

    def test_curve_segment(self):
        for time in self.time_samples[:-1]:
            self.assertEqual(self.composite_curve.curve_segment(time), int(time))
        self.assertEqual(self.composite_curve.curve_segment(self.final_time), self.n_curves - 1)

    def test_call(self):
        for time in self.time_samples:
            value = self.composite_curve(time)
            i = self.composite_curve.curve_segment(time)
            curve = self.composite_curve.curves[i]
            np.testing.assert_array_almost_equal(value, curve(time))

    def test_initial_final_point(self):
        initial_value = self.composite_curve(self.initial_time)
        initial_point = self.composite_curve.initial_point()
        np.testing.assert_array_almost_equal(initial_value, initial_point)
        final_value = self.composite_curve(self.final_time)
        final_point = self.composite_curve.final_point()
        np.testing.assert_array_almost_equal(final_value, final_point)

    def test_iter(self):
        for curve in self.composite_curve:
            self.assertEqual(curve.duration, 1)

    def test_getitem(self):
        for i in range(self.n_curves):
            curve = self.composite_curve[i]
            self.assertEqual(curve.duration, 1)

    def test_len(self):
        self.assertEqual(len(self.composite_curve), self.n_curves)

    def test_scalar_mul(self):
        c = 3.66
        prod_1 = self.composite_curve * c
        prod_2 = c * self.composite_curve
        for time in self.time_samples:
            value = self.composite_curve(time) * c
            np.testing.assert_array_almost_equal(prod_1(time), value)
            np.testing.assert_array_almost_equal(prod_2(time), value)
        
    def test_elementwise_mul(self):
        prod = self.composite_curve_1 * self.composite_curve_2
        for time in self.time_samples:
            value = self.composite_curve_1(time) * self.composite_curve_2(time)
            np.testing.assert_array_almost_equal(prod(time), value)

    def test_elevate_degree(self):
        composite_curve = self.composite_curve.elevate_degree(11)
        for time in self.time_samples:
            np.testing.assert_array_almost_equal(self.composite_curve(time), composite_curve(time))

    def test_scalar_add_sub(self):
        c = 3.66
        sum_1 = self.composite_curve + c
        sum_2 = c + self.composite_curve
        sub_1 = self.composite_curve - c
        sub_2 = c - self.composite_curve
        for time in self.time_samples:
            value = self.composite_curve(time) + c
            np.testing.assert_array_almost_equal(sum_1(time), value)
            np.testing.assert_array_almost_equal(sum_2(time), value)
            value -= 2 * c
            np.testing.assert_array_almost_equal(sub_1(time), value)
            np.testing.assert_array_almost_equal(sub_2(time), -value)

    def test_elementwise_add_sub(self):
        sum = self.composite_curve_1 + self.composite_curve_2
        sub = self.composite_curve_1 - self.composite_curve_2
        for time in self.time_samples:
            value = self.composite_curve_1(time) + self.composite_curve_2(time)
            np.testing.assert_array_almost_equal(sum(time), value)
            value -= 2 * self.composite_curve_2(time)
            np.testing.assert_array_almost_equal(sub(time), value)

    def test_neg(self):
        neg = - self.composite_curve
        for time in self.time_samples:
            np.testing.assert_array_almost_equal(neg(time), -self.composite_curve(time))

    def test_derivative(self):
        der = self.composite_curve.derivative()
        time_step = 1e-9
        for time in np.linspace(self.initial_time, self.final_time - time_step):
            value = (self.composite_curve(time + time_step) - self.composite_curve(time)) / time_step
            np.testing.assert_array_almost_equal(der(time), value)

    def test_knot_points(self):
        for i, point in enumerate(self.composite_curve.knot_points()):
            np.testing.assert_array_almost_equal(point, self.composite_curve(i))

    def test_durations(self):
        durations = list(self.composite_curve.durations())
        self.assertEqual(durations, [1] * self.n_curves)

    def test_concatenate(self):
        conc = self.composite_curve_1.concatenate(self.composite_curve_2)
        for time in self.time_samples * 2:
            if time < self.final_time:
                value = self.composite_curve_1(time)
                np.testing.assert_array_almost_equal(conc(time), value)
            elif time > self.final_time:
                value = self.composite_curve_2(time - self.final_time)
                np.testing.assert_array_almost_equal(conc(time), value)    
                
    def test_l2_squared(self):
        n_samples = 5000
        times = np.linspace(self.initial_time, self.final_time, n_samples)
        squared_norm = lambda time: np.linalg.norm(self.composite_curve(time)) ** 2
        values = [squared_norm(time) for time in times]
        integral = np.trapezoid(values, times)
        self.assertAlmostEqual(self.composite_curve.l2_squared(), integral, places=4)

if __name__ == '__main__':
    unittest.main()