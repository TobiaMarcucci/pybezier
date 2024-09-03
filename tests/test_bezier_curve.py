import unittest
import numpy as np

from pybezier.bezier_curve import BezierCurve

class TestBezierCurve(unittest.TestCase):

    def test_init(self):
        np.random.seed(0)
        points = np.random.rand(11, 3)
        initial_time = .55
        final_time = 3.12
        curve = BezierCurve(points, initial_time, final_time)
        np.testing.assert_equal(curve.points, points)
        self.assertEqual(curve.initial_time, initial_time)
        self.assertEqual(curve.final_time, final_time)
        self.assertEqual(curve.duration, final_time - initial_time)
        self.assertEqual(curve.degree, points.shape[0] - 1)
        self.assertEqual(curve.dimension, points.shape[1])
        self.assertRaises(ValueError, BezierCurve, points, final_time, initial_time)

    def test_init_deafult(self):
        np.random.seed(0)
        points = np.random.rand(11, 3)
        curve = BezierCurve(points)
        self.assertEqual(curve.initial_time, 0)
        self.assertEqual(curve.final_time, 1)

    def test_initial_final_point(self):
        np.random.seed(0)
        points = np.random.rand(11, 3)
        curve = BezierCurve(points)
        np.testing.assert_array_almost_equal(curve(0), points[0])
        np.testing.assert_array_almost_equal(curve(1), points[-1])

    def test_berstein(self):
        np.random.seed(0)
        n_points = 4
        points = np.random.rand(n_points, 5)
        initial_time = 0
        final_time = 1
        curve = BezierCurve(points, initial_time, final_time)
        for time in np.linspace(initial_time, final_time):
            values = [curve._berstein(time, n) for n in range(n_points)]
            # tests partition of unity
            self.assertTrue(min(values) >= 0)
            self.assertAlmostEqual(sum(values), 1)

    def test_assert_same_times(self):
        np.random.seed(0)
        points = np.random.rand(11, 3)
        initial_time_1 = .55
        initial_time_2 = .3
        final_time_1 = 3.12
        final_time_2 = 1.
        curve_1 = BezierCurve(points, initial_time_1, final_time_1)
        curve_2 = BezierCurve(points, initial_time_2, final_time_2)
        curve_1._assert_same_times(curve_1)
        self.assertRaises(ValueError, curve_1._assert_same_times, curve_2)

    def test_call(self):
        points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        initial_time = 0
        final_time = 3
        curve = BezierCurve(points, initial_time, final_time)
        np.testing.assert_allclose(curve(initial_time), points[0])
        np.testing.assert_allclose(curve(final_time), points[-1])
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
        np.random.seed(0)
        c = 3.66
        points = np.random.rand(7, 3)
        curve = BezierCurve(points)
        prod_1 = curve * c
        prod_2 = c * curve
        for time in np.linspace(0, 1):
            value = curve(time) * c
            np.testing.assert_array_almost_equal(prod_1(time), value)
            np.testing.assert_array_almost_equal(prod_2(time), value)
        
    def test_elementwise_mul(self):
        np.random.seed(0)
        points_1 = np.random.rand(3, 3)
        points_2 = np.random.rand(5, 3)
        initial_time = .55
        final_time = 2.3
        curve_1 = BezierCurve(points_1, initial_time, final_time)
        curve_2 = BezierCurve(points_2, initial_time, final_time)
        prod = curve_1 * curve_2
        for time in np.linspace(initial_time, final_time):
            np.testing.assert_array_almost_equal(prod(time), curve_1(time) * curve_2(time))

    def test_elevate_degree(self):
        np.random.seed(0)
        points = np.random.rand(3, 3)
        curve_1 = BezierCurve(points)
        curve_2 = curve_1.elevate_degree(7)
        for time in np.linspace(0, 1):
            np.testing.assert_array_almost_equal(curve_1(time), curve_2(time))

    def test_scalar_add_sub(self):
        points_1 = np.random.rand(3, 3)
        points_2 = np.random.rand(7, 3)
        curve_1 = BezierCurve(points_1)
        curve_2 = BezierCurve(points_2)
        curve_sum = curve_1 + curve_2
        curve_sub = curve_1 - curve_2
        for time in np.linspace(0, 1):
            np.testing.assert_array_almost_equal(curve_sum(time), curve_1(time) + curve_2(time))
            np.testing.assert_array_almost_equal(curve_sub(time), curve_1(time) - curve_2(time))

    def test_elementwise_add_sub(self):
        np.random.seed(0)
        c = 3.66
        points = np.random.rand(7, 3)
        curve = BezierCurve(points)
        curve_sum_1 = curve + c
        curve_sum_2 = c + curve
        curve_sub_1 = curve - c
        curve_sub_2 = c - curve
        for time in np.linspace(0, 1):
            value = curve(time) + c
            np.testing.assert_array_almost_equal(curve_sum_1(time), value)
            np.testing.assert_array_almost_equal(curve_sum_2(time), value)
            value = curve(time) - c
            np.testing.assert_array_almost_equal(curve_sub_1(time), value)
            np.testing.assert_array_almost_equal(curve_sub_2(time), -value)

    def test_neg(self):
        np.random.seed(0)
        points = np.random.rand(7, 3)
        curve = BezierCurve(points)
        curve_neg = - curve
        for time in np.linspace(0, 1):
            np.testing.assert_array_almost_equal(curve_neg(time), -curve(time))

    def test_derivative(self):
        np.random.seed(0)
        points = np.random.rand(7, 3)
        curve = BezierCurve(points)
        derivative = curve.derivative()
        time_step = 1e-9
        for time in np.linspace(0, 1 - time_step):
            value = (curve(time + time_step) - curve(time)) / time_step
            np.testing.assert_array_almost_equal(derivative(time), value)

    def test_split(self):
        np.random.seed(0)
        points = np.random.rand(7, 3)
        curve = BezierCurve(points)
        split_time = 0.5
        curve_1, curve_2 = curve.split(split_time)
        for time in np.linspace(0, 1):
            if time < split_time:
                np.testing.assert_array_almost_equal(curve(time), curve_1(time))
            elif time > split_time:
                np.testing.assert_array_almost_equal(curve(time), curve_2(time))

    def test_l2_squared(self):
        points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        initial_time = .55
        final_time = 3.12
        curve = BezierCurve(points, initial_time, final_time)
        times = np.linspace(initial_time, final_time, 100)
        values = [curve(time).dot(curve(time)) for time in times]
        integral = np.trapezoid(values, times)
        self.assertAlmostEqual(curve.l2_squared(), integral)

    def test_integral_of_convex(self):
        points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        initial_time = .55
        final_time = 3.12
        curve = BezierCurve(points, initial_time, final_time)
        f = lambda point: point.dot(point)
        self.assertTrue(curve.integral_of_convex(f) >= curve.l2_squared())
        # upper bound for curve lenght is equal to distance of control points
        derivative = curve.derivative()
        self.assertAlmostEqual(derivative.integral_of_convex(np.linalg.norm), 3)
        