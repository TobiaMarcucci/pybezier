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
        self.assertEqual(self.composite_curve.transition_times, list(range(self.n_curves + 1)))

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
        initial_point = self.composite_curve.initial_point
        np.testing.assert_array_almost_equal(initial_value, initial_point)
        final_value = self.composite_curve(self.final_time)
        final_point = self.composite_curve.final_point
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
        derivative = self.composite_curve.derivative()
        time_step = 1e-7
        for time in np.linspace(self.initial_time, self.final_time - time_step):
            numerical_derivative = (self.composite_curve(time + time_step) - self.composite_curve(time)) / time_step
            np.testing.assert_array_almost_equal(derivative(time), numerical_derivative)

    def test_integral(self):
        initial_conditions = [None, np.ones(self.composite_curve.dimension)]
        for initial_condition in initial_conditions:
            integral = self.composite_curve.integral(initial_condition)
            value = integral(self.initial_time)
            target_value = 0 if initial_condition is None else initial_condition
            np.testing.assert_array_almost_equal(value, target_value)
            time_step = 1e-3
            for time in np.linspace(self.initial_time, self.final_time - time_step):
                value = self.composite_curve(time + time_step / 2)
                target_value = (integral(time + time_step) - integral(time)) / time_step
                np.testing.assert_array_almost_equal(value, target_value)

    def test_domain_split(self):

        # split at initial time
        composite_curve_1, composite_curve_2 = self.composite_curve.domain_split(self.initial_time)
        self.assertTrue(composite_curve_1 is None)
        for curve_1, curve_2 in zip(self.composite_curve, composite_curve_2):
            np.testing.assert_array_equal(curve_1.points, curve_2.points)
            self.assertEqual(curve_1.initial_time, curve_2.initial_time)
            self.assertEqual(curve_1.final_time, curve_2.final_time)

        # split at terminal time
        composite_curve_1, composite_curve_2 = self.composite_curve.domain_split(self.final_time)
        self.assertTrue(composite_curve_2 is None)
        for curve_1, curve_2 in zip(self.composite_curve, composite_curve_1):
            np.testing.assert_array_equal(curve_1.points, curve_2.points)
            self.assertEqual(curve_1.initial_time, curve_2.initial_time)
            self.assertEqual(curve_1.final_time, curve_2.final_time)

        # split at transition times
        for split_time in self.composite_curve.transition_times[1:-1]:
            composite_curve_1, composite_curve_2 = self.composite_curve.domain_split(split_time)
            composite_curve = composite_curve_1.concatenate(composite_curve_2)
            for curve_1, curve_2 in zip(self.composite_curve, composite_curve):
                np.testing.assert_array_equal(curve_1.points, curve_2.points)
                self.assertEqual(curve_1.initial_time, curve_2.initial_time)
                self.assertEqual(curve_1.final_time, curve_2.final_time)

        # split at internal time
        for split_time in np.linspace(self.initial_time + 1e-3, self.final_time - 1e-3):
            composite_curve_1, composite_curve_2 = self.composite_curve.domain_split(split_time)
            for time in self.time_samples:
                if time < split_time:
                    np.testing.assert_array_almost_equal(self.composite_curve(time), composite_curve_1(time))
                elif time > split_time:
                    np.testing.assert_array_almost_equal(self.composite_curve(time), composite_curve_2(time))

        # split outside domain
        with self.assertRaises(ValueError):
            self.assertRaises(self.composite_curve.domain_split(self.initial_time - .1))
        with self.assertRaises(ValueError):
            self.assertRaises(self.composite_curve.domain_split(self.final_time + .1))

    def test_time_shift(self):
        t = .234
        shifted_composite_curve = self.composite_curve.time_shift(t)
        self.assertEqual(len(shifted_composite_curve), len(self.composite_curve))
        for curve, shifted_curve in zip(self.composite_curve, shifted_composite_curve):
            np.testing.assert_array_equal(curve.points, shifted_curve.points)
            self.assertEqual(curve.initial_time + t, shifted_curve.initial_time)
            self.assertEqual(curve.final_time + t, shifted_curve.final_time)

    def test_transition_points(self):
        for i, point in enumerate(self.composite_curve.transition_points()):
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

    def test_integral_of_convex_function(self):
        f = lambda point: point.dot(point)
        value = self.composite_curve.l2_squared()
        upper_bound = self.composite_curve.integral_of_convex_function(f)
        self.assertTrue(value <= upper_bound)
        # upper bound for curve lenght is equal to distance of control points
        diff_points = np.vstack([curve.points[:-1] - curve.points[1:] for curve in self.composite_curve])
        value = sum(np.linalg.norm(diff_points, axis=1))
        derivative = self.composite_curve.derivative()
        upper_bound = derivative.integral_of_convex_function(np.linalg.norm)
        self.assertAlmostEqual(value, upper_bound)

if __name__ == '__main__':
    unittest.main()