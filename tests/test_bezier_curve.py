import unittest
import numpy as np
from pybezier.bezier_curve import BezierCurve

class TestBezierCurve(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.points = np.random.rand(5, 3)
        self.initial_time = .55
        self.final_time = 3.12
        self.curve = BezierCurve(self.points, self.initial_time, self.final_time)
        self.curve_1 = self.curve
        self.points_2 = np.random.rand(7, 3)
        self.curve_2 = BezierCurve(self.points_2, self.initial_time, self.final_time)
        self.time_samples = np.linspace(self.initial_time, self.final_time)

    def test_init(self):
        np.testing.assert_equal(self.curve.points, self.points)
        self.assertEqual(self.curve.initial_time, self.initial_time)
        self.assertEqual(self.curve.final_time, self.final_time)
        self.assertEqual(self.curve.duration, self.final_time - self.initial_time)
        self.assertEqual(self.curve.degree, self.points.shape[0] - 1)
        self.assertEqual(self.curve.dimension, self.points.shape[1])
        self.assertRaises(ValueError, BezierCurve, self.points, self.final_time, self.initial_time)

    def test_init_deafult(self):
        curve = BezierCurve(self.points)
        self.assertEqual(curve.initial_time, 0)
        self.assertEqual(curve.final_time, 1)

    def test_initial_final_point(self):
        np.testing.assert_array_almost_equal(self.curve(self.initial_time), self.curve.initial_point)
        np.testing.assert_array_almost_equal(self.curve(self.final_time), self.curve.final_point)

    def test_berstein(self):
        for time in self.time_samples:
            values = [self.curve._berstein(time, n) for n in range(self.curve.degree + 1)]
            # tests partition of unity
            self.assertTrue(min(values) >= 0)
            self.assertAlmostEqual(sum(values), 1)

    def test_assert_same_times(self):
        self.curve_1._assert_same_times(self.curve_2)
        curve_3 = BezierCurve(self.points)
        self.assertRaises(ValueError, self.curve_1._assert_same_times, curve_3)

    def test_call(self):
        points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        initial_time = 0
        final_time = 3
        curve = BezierCurve(points, initial_time, final_time)
        p1 = curve(1)
        p2 = curve(2)
        # tests convex-hull property
        np.testing.assert_array_less(p1, points[2])
        np.testing.assert_array_less(p2, points[2])
        np.testing.assert_array_less(points[0], p1)
        np.testing.assert_array_less(points[0], p2)
        # tests symmetry
        self.assertAlmostEqual(p1[0], 1 - p2[0])
        self.assertAlmostEqual(p1[1], p2[1])

    def test_scalar_mul(self):
        c = 3.66
        prod_1 = self.curve * c
        prod_2 = c * self.curve
        for time in self.time_samples:
            value = self.curve(time) * c
            np.testing.assert_array_almost_equal(prod_1(time), value)
            np.testing.assert_array_almost_equal(prod_2(time), value)
        
    def test_elementwise_mul(self):
        prod = self.curve_1 * self.curve_2
        for time in self.time_samples:
            np.testing.assert_array_almost_equal(prod(time), self.curve_1(time) * self.curve_2(time))

    def test_elevate_degree(self):
        curve = self.curve.elevate_degree(11)
        for time in self.time_samples:
            np.testing.assert_array_almost_equal(self.curve(time), curve(time))

    def test_scalar_add_sub(self):
        c = 3.66
        sum_1 = self.curve + c
        sum_2 = c + self.curve
        sub_1 = self.curve - c
        sub_2 = c - self.curve
        for time in self.time_samples:
            value = self.curve(time) + c
            np.testing.assert_array_almost_equal(sum_1(time), value)
            np.testing.assert_array_almost_equal(sum_2(time), value)
            value -= 2 * c
            np.testing.assert_array_almost_equal(sub_1(time), value)
            np.testing.assert_array_almost_equal(sub_2(time), -value)

    def test_elementwise_add_sub(self):
        sum = self.curve_1 + self.curve_2
        sub = self.curve_1 - self.curve_2
        for time in self.time_samples:
            value = self.curve_1(time) + self.curve_2(time)
            np.testing.assert_array_almost_equal(sum(time), value)
            value -= 2 * self.curve_2(time)
            np.testing.assert_array_almost_equal(sub(time), value)

    def test_neg(self):
        neg = - self.curve
        for time in self.time_samples:
            np.testing.assert_array_almost_equal(neg(time), -self.curve(time))

    def test_derivative(self):
        derivative = self.curve.derivative()
        time_step = 1e-6
        for time in np.linspace(self.initial_time, self.final_time - time_step):
            numerical_derivative = (self.curve(time + time_step) - self.curve(time)) / time_step
            np.testing.assert_array_almost_equal(derivative(time), numerical_derivative)

    def test_integral(self):
        initial_conditions = [None, np.ones(self.curve.dimension)]
        for initial_condition in initial_conditions:
            integral = self.curve.integral(initial_condition)
            value = integral(self.initial_time)
            target_value = 0 if initial_condition is None else initial_condition
            np.testing.assert_array_almost_equal(value, target_value)
            time_step = 1e-3
            for time in np.linspace(self.initial_time, self.final_time - time_step):
                value = self.curve(time + time_step / 2)
                target_value = (integral(time + time_step) - integral(time)) / time_step
                np.testing.assert_array_almost_equal(value, target_value)

    def test_domain_split(self):

        # split at initial time
        curve_1, curve_2 = self.curve.domain_split(self.initial_time)
        self.assertTrue(curve_1 is None)
        np.testing.assert_array_equal(self.curve.points, curve_2.points)
        self.assertEqual(self.initial_time, curve_2.initial_time)
        self.assertEqual(self.final_time, curve_2.final_time)

        # split at final time
        curve_1, curve_2 = self.curve.domain_split(self.final_time)
        self.assertTrue(curve_2 is None)
        np.testing.assert_array_equal(self.curve.points, curve_1.points)
        self.assertEqual(self.initial_time, curve_1.initial_time)
        self.assertEqual(self.final_time, curve_1.final_time)

        # split at internal time
        split_time = (self.initial_time + self.final_time) / 2
        curve_1, curve_2 = self.curve.domain_split(split_time)
        for time in self.time_samples:
            if time < split_time:
                np.testing.assert_array_almost_equal(self.curve(time), curve_1(time))
            elif time > split_time:
                np.testing.assert_array_almost_equal(self.curve(time), curve_2(time))

        # split outside domain
        with self.assertRaises(ValueError):
            self.assertRaises(self.curve.domain_split(self.initial_time - .1))
        with self.assertRaises(ValueError):
            self.assertRaises(self.curve.domain_split(self.final_time + .1))

    def test_time_shift(self):
        t = .33
        shifted_curve = self.curve.time_shift(t)
        np.testing.assert_array_equal(self.curve.points, shifted_curve.points)
        self.assertEqual(self.initial_time + t, shifted_curve.initial_time)
        self.assertEqual(self.final_time + t, shifted_curve.final_time)

    def test_l2_squared(self):
        n_samples = 5000
        times = np.linspace(self.initial_time, self.final_time, n_samples)
        values = [self.curve(time).dot(self.curve(time)) for time in times]
        integral = np.trapezoid(values, times)
        self.assertAlmostEqual(self.curve.l2_squared(), integral)

    def test_integral_of_convex_function(self):
        f = lambda point: point.dot(point)
        value = self.curve.l2_squared()
        upper_bound = self.curve.integral_of_convex_function(f)
        self.assertTrue(value <= upper_bound)
        # upper bound for curve lenght is equal to distance of control points
        diff_points = self.points[:-1] - self.points[1:]
        value = sum(np.linalg.norm(diff_points, axis=1))
        derivative = self.curve.derivative()
        upper_bound = derivative.integral_of_convex_function(np.linalg.norm)
        self.assertAlmostEqual(value, upper_bound)
        
if __name__ == '__main__':
    unittest.main()