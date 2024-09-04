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
        np.testing.assert_array_almost_equal(self.curve(self.initial_time), self.points[0])
        np.testing.assert_array_almost_equal(self.curve(self.final_time), self.points[-1])

    def test_berstein(self):
        for time in np.linspace(self.initial_time, self.final_time):
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
        np.random.seed(0)
        c = 3.66
        prod_1 = self.curve * c
        prod_2 = c * self.curve
        for time in np.linspace(self.initial_time, self.final_time):
            value = self.curve(time) * c
            np.testing.assert_array_almost_equal(prod_1(time), value)
            np.testing.assert_array_almost_equal(prod_2(time), value)
        
    def test_elementwise_mul(self):
        prod = self.curve_1 * self.curve_2
        for time in np.linspace(self.initial_time, self.final_time):
            np.testing.assert_array_almost_equal(prod(time), self.curve_1(time) * self.curve_2(time))

    def test_elevate_degree(self):
        curve = self.curve.elevate_degree(11)
        for time in np.linspace(self.initial_time, self.final_time):
            np.testing.assert_array_almost_equal(self.curve(time), curve(time))

    def test_scalar_add_sub(self):
        c = 3.66
        curve_sum_1 = self.curve + c
        curve_sum_2 = c + self.curve
        curve_sub_1 = self.curve - c
        curve_sub_2 = c - self.curve
        for time in np.linspace(self.initial_time, self.final_time):
            value = self.curve(time) + c
            np.testing.assert_array_almost_equal(curve_sum_1(time), value)
            np.testing.assert_array_almost_equal(curve_sum_2(time), value)
            value -= 2 * c
            np.testing.assert_array_almost_equal(curve_sub_1(time), value)
            np.testing.assert_array_almost_equal(curve_sub_2(time), -value)

    def test_elementwise_add_sub(self):
        curve_sum = self.curve_1 + self.curve_2
        curve_sub = self.curve_1 - self.curve_2
        for time in np.linspace(self.initial_time, self.final_time):
            value = self.curve_1(time) + self.curve_2(time)
            np.testing.assert_array_almost_equal(curve_sum(time), value)
            value -= 2 * self.curve_2(time)
            np.testing.assert_array_almost_equal(curve_sub(time), value)

    def test_neg(self):
        curve = - self.curve
        for time in np.linspace(0, 1):
            np.testing.assert_array_almost_equal(curve(time), -self.curve(time))

    def test_derivative(self):
        derivative = self.curve.derivative()
        time_step = 1e-9
        for time in np.linspace(self.initial_time, self.final_time - time_step):
            value = (self.curve(time + time_step) - self.curve(time)) / time_step
            np.testing.assert_array_almost_equal(derivative(time), value)

    def test_split(self):
        split_time = (self.initial_time + self.final_time) / 2
        curve_1, curve_2 = self.curve.split(split_time)
        for time in np.linspace(self.initial_time, self.final_time):
            if time < split_time:
                np.testing.assert_array_almost_equal(self.curve(time), curve_1(time))
            elif time > split_time:
                np.testing.assert_array_almost_equal(self.curve(time), curve_2(time))

    def test_l2_squared(self):
        n_samples = 5000
        times = np.linspace(self.initial_time, self.final_time, n_samples)
        values = [self.curve(time).dot(self.curve(time)) for time in times]
        integral = np.trapezoid(values, times)
        self.assertAlmostEqual(self.curve.l2_squared(), integral)

    def test_integral_of_convex(self):
        f = lambda point: point.dot(point)
        self.assertTrue(self.curve.integral_of_convex(f) >= self.curve.l2_squared())
        # upper bound for curve lenght is equal to distance of control points
        derivative = self.curve.derivative()
        value = sum(np.linalg.norm(y - x) for x, y in zip(self.points[:-1], self.points[1:]))
        self.assertAlmostEqual(derivative.integral_of_convex(np.linalg.norm), value)
        
if __name__ == '__main__':
    unittest.main()