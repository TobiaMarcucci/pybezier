import unittest
from pybezier.binomial import binomial

class TestBinomial(unittest.TestCase):

    def test_valid_inputs(self):
        self.assertEqual(binomial(4, 2), 6)
        self.assertEqual(binomial(8, 3), 56)
        self.assertEqual(binomial(8, 8), 1)
        self.assertEqual(binomial(9, 5), 126)
        self.assertEqual(binomial(9, 7), 36)
        self.assertEqual(binomial(11, 0), 1)

    def test_invalid_inputs(self):
        self.assertRaises(ValueError, binomial, 4, .5)
        self.assertRaises(ValueError, binomial, 4.5, 2)
        self.assertRaises(ValueError, binomial, 4, 5)
        self.assertRaises(ValueError, binomial, 4, -1)

if __name__ == '__main__':
    unittest.main()