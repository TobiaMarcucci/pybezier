import unittest
import numpy as np
from pybezier.bezier import BezierCurve

class TestBezierCurve(unittest.TestCase):

    def setUp(self):
        self.initial_time = .55
        self.final_time = 3.12
        self.time_samples = np.linspace(self.initial_time, self.final_time)
        # scalar curve
        np.random.seed(0)
        self.scalar_points = np.random.rand(6)
        self.scalar_curve = BezierCurve(self.scalar_points, self.initial_time, self.final_time)
        # vector curve
        np.random.seed(0)
        self.vec_points = np.random.rand(7, 2)
        self.vec_curve = BezierCurve(self.vec_points, self.initial_time, self.final_time)
        # matrix curve
        np.random.seed(0)
        self.mat_points = np.random.rand(7, 5, 3)
        self.mat_curve = BezierCurve(self.mat_points, self.initial_time, self.final_time)
        # all curves
        self.points = [self.scalar_points, self.vec_points, self.mat_points]
        self.curves = [self.scalar_curve, self.vec_curve, self.mat_curve]

    def test_init(self):
        for curve, points in zip(self.curves, self.points):
            np.testing.assert_equal(curve.points, points)
            self.assertEqual(curve.initial_time, self.initial_time)
            self.assertEqual(curve.final_time, self.final_time)
            self.assertRaises(ValueError, BezierCurve, points, self.final_time, self.initial_time)

    def test_deafult_init(self):
        for points in self.points:
            curve = BezierCurve(points)
            self.assertEqual(curve.initial_time, 0)
            self.assertEqual(curve.final_time, 1)

    def test_degree(self):
        for curve in self.curves:
            self.assertEqual(curve.degree, len(curve.points) - 1)

    def test_shape(self):
        for curve in self.curves:
            self.assertEqual(curve.shape, curve.points[0].shape)

    def test_dimension(self):
        for curve in self.curves:
            self.assertEqual(curve.dimension, curve.points[0].size)

    def test_duration(self):
        for curve in self.curves:
            self.assertEqual(curve.duration, self.final_time - self.initial_time)

    def test_initial_point(self):
        for curve in self.curves:
            np.testing.assert_array_almost_equal(curve.initial_point, curve(self.initial_time))

    def test_final_point(self):
        for curve in self.curves:
            np.testing.assert_array_almost_equal(curve.final_point, curve(self.final_time))

    def test_berstein(self):
        for time in self.time_samples:
            values = [self.vec_curve._berstein(time, n) for n in range(self.vec_curve.degree + 1)]
            # partition of unity
            self.assertTrue(min(values) >= 0)
            self.assertAlmostEqual(sum(values), 1)

    def test_check_same_times(self):
        self.vec_curve._check_same_times(self.mat_curve)
        curve = BezierCurve(self.vec_points)
        self.assertRaises(ValueError, self.vec_curve._check_same_times, curve)

    def test_call(self):
        scalar_points = np.array([0, 1, 2, 3])
        vec_points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        mat_points = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]])
        for points in [scalar_points, vec_points, mat_points]:
            curve = BezierCurve(points)
            for time in np.linspace(0, 1):
                point = points[0] + time * points[-1]
                np.testing.assert_array_almost_equal(point, curve(time))

    def test_scalar_mul_and_rmul(self):
        c = 3.66
        for curve in self.curves:
            prod_1 = curve * c
            prod_2 = c * curve
            for time in self.time_samples:
                value = curve(time) * c
                np.testing.assert_array_almost_equal(prod_1(time), value)
                np.testing.assert_array_almost_equal(prod_2(time), value)
        
    def test_elementwise_mul(self):
        for curve in self.curves:
            prod = curve * curve
            for time in self.time_samples:
                value = curve(time) * curve(time)
                np.testing.assert_array_almost_equal(prod(time), value)
        with self.assertRaises(ValueError):
            self.mat_curve * self.vec_curve

    def test_imul(self):
        c = 3.66
        curve = BezierCurve(self.vec_points)
        curve *= c
        for point_1, point_2 in zip(curve.points, self.vec_points):
            np.testing.assert_array_almost_equal(point_1, point_2 * c)

    def test_matmul(self):
        curve_1 = self.vec_curve @ self.vec_curve
        curve_2 = self.vec_curve * self.vec_curve
        for point_1, point_2 in zip(curve_1.points, curve_2.points):
            np.testing.assert_array_almost_equal(point_1, sum(point_2))
        with self.assertRaises(TypeError):
            self.scalar_curve @ self.scalar_curve
        with self.assertRaises(ValueError):
            self.mat_curve @ self.vec_curve

    def test_scalar_add_sub(self):
        c = 3.66
        for curve in self.curves:
            sum_1 = curve + c
            sum_2 = c + curve
            sub_1 = curve - c
            sub_2 = c - curve
            for time in self.time_samples:
                np.testing.assert_array_almost_equal(sum_1(time), curve(time) + c)
                np.testing.assert_array_almost_equal(sum_2(time), c + curve(time))
                np.testing.assert_array_almost_equal(sub_1(time), curve(time) - c)
                np.testing.assert_array_almost_equal(sub_2(time), c - curve(time))

    def test_elementwise_add_sub(self):
        for curve in self.curves:
            sum = curve + curve
            sub = curve - curve
            for time in self.time_samples:
                np.testing.assert_array_almost_equal(sum(time), 2 * curve(time))
                np.testing.assert_array_almost_equal(sub(time), 0)

    def test_iadd_and_isub(self):
        for points, curve in zip(self.points, self.curves):
            curve_2 = BezierCurve(points, self.initial_time, self.final_time)
            curve_2 += curve
            for time in self.time_samples:
                np.testing.assert_array_almost_equal(curve_2(time), 2 * curve(time))
            curve_2 -= curve
            for time in self.time_samples:
                np.testing.assert_array_almost_equal(curve_2(time), curve(time))

    def test_neg(self):
        for curve in self.curves:
            neg = - curve
            for time in self.time_samples:
                np.testing.assert_array_almost_equal(neg(time), -curve(time))

    def test_elevate_degree(self):
        for curve_1 in self.curves:
            curve_2 = curve_1.elevate_degree(11)
            for time in self.time_samples:
                np.testing.assert_array_almost_equal(curve_1(time), curve_2(time))

    def test_derivative(self):
        time_step = 1e-6
        for curve in self.curves:
            derivative = curve.derivative()
            for time in np.linspace(self.initial_time, self.final_time - time_step):
                numerical_derivative = (curve(time + time_step) - curve(time)) / time_step
                np.testing.assert_array_almost_equal(derivative(time), numerical_derivative, decimal=5)

    def test_integral(self):
        time_step = 1e-3
        for curve in self.curves:
            for initial_condition in [None, np.ones(curve.shape)]:
                integral = curve.integral(initial_condition)
                value = integral(self.initial_time)
                if initial_condition is None:
                    np.testing.assert_array_almost_equal(value, 0)
                else:
                    np.testing.assert_array_almost_equal(value, initial_condition)
                for time in np.linspace(self.initial_time, self.final_time - time_step):
                    value = curve(time + time_step / 2)
                    target_value = (integral(time + time_step) - integral(time)) / time_step
                    np.testing.assert_array_almost_equal(value, target_value)

    def test_split_domain(self):
        for curve in self.curves:

            # split at initial time
            curve_1, curve_2 = curve.split_domain(self.initial_time)
            self.assertTrue(curve_1 is None)
            np.testing.assert_array_equal(curve.points, curve_2.points)
            self.assertEqual(self.initial_time, curve_2.initial_time)
            self.assertEqual(self.final_time, curve_2.final_time)

            # split at final time
            curve_1, curve_2 = curve.split_domain(self.final_time)
            self.assertTrue(curve_2 is None)
            np.testing.assert_array_equal(curve.points, curve_1.points)
            self.assertEqual(self.initial_time, curve_1.initial_time)
            self.assertEqual(self.final_time, curve_1.final_time)

            # split at internal time
            split_time = (self.initial_time + self.final_time) / 2
            curve_1, curve_2 = curve.split_domain(split_time)
            for time in self.time_samples:
                if time < split_time:
                    np.testing.assert_array_almost_equal(curve(time), curve_1(time))
                elif time > split_time:
                    np.testing.assert_array_almost_equal(curve(time), curve_2(time))

            # split outside domain
            with self.assertRaises(ValueError):
                curve.split_domain(self.initial_time - .1)
            with self.assertRaises(ValueError):
                curve.split_domain(self.final_time + .1)

    def test_shift_domain(self):
        t = .33
        for curve in self.curves:
            shifted_curve = curve.shift_domain(t)
            np.testing.assert_array_equal(curve.points, shifted_curve.points)
            self.assertEqual(self.initial_time + t, shifted_curve.initial_time)
            self.assertEqual(self.final_time + t, shifted_curve.final_time)

    def test_squared_l2_norm(self):
        n_samples = 1000
        times = np.linspace(self.initial_time, self.final_time, n_samples)
        values = [np.linalg.norm(self.vec_curve(time)) ** 2 for time in times]
        integral = np.trapz(values, times)
        self.assertAlmostEqual(self.vec_curve.squared_l2_norm(), integral, places=4)

    def test_integral_of_convex_function(self):
        f = lambda point: point.dot(point)
        value = self.vec_curve.squared_l2_norm()
        upper_bound = self.vec_curve.integral_of_convex_function(f)
        self.assertTrue(value <= upper_bound)
        # upper bound for curve lenght is equal to distance of control points
        diff_points = self.vec_points[:-1] - self.vec_points[1:]
        value = sum(np.linalg.norm(diff_points, axis=1))
        derivative = self.vec_curve.derivative()
        upper_bound = derivative.integral_of_convex_function(np.linalg.norm)
        self.assertAlmostEqual(value, upper_bound)
        
if __name__ == '__main__':
    unittest.main()
