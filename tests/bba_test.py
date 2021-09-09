import unittest
from bact_analysis.bba import calc


class Test10_AngleToOffset(unittest.TestCase):
    """Checking that it is just a multiplication"""

    def test10_ZeroAngle(self):
        offset = calc.angle_to_offset(1, 1, 1, 0)
        self.assertAlmostEqual(offset, 0)

    def test11_AngleOne(self):
        offset = calc.angle_to_offset(1, 1, 1, 1)
        self.assertAlmostEqual(offset, 1)

    def test11_AngleOneLength2(self):
        offset = calc.angle_to_offset(1, 2, 1, 1)
        self.assertAlmostEqual(offset, 1 / 2)

    def test11_AngleNegOneLength2(self):
        offset = calc.angle_to_offset(1, 2, 1, -1)
        self.assertAlmostEqual(offset, -1 / 2)

    def test11_AngleNegOneLength4(self):
        offset = calc.angle_to_offset(1, 4, 1, -1)
        self.assertAlmostEqual(offset, -1 / 4)

    def test11_Transferfunction(self):
        offset = calc.angle_to_offset(2 / 5, 1, 1, 1)
        self.assertAlmostEqual(offset, 5 / 2)


class Test20_AngleToOffsetMagnetInfo(unittest.TestCase):
    def test10_TransferFunction(self):
        mi = calc.MagnetInfo(tf=2/5, length=1, polarity=1)
        offset = mi.angle_to_offset(1)
        self.assertAlmostEqual(offset, 5 / 2)

    def test11_TransferFunctionNetPolarity(self):
        mi = calc.MagnetInfo(tf=2/5, length=1, polarity=-1)
        offset = mi.angle_to_offset(1)
        self.assertAlmostEqual(offset, -5 / 2)


if __name__ == "__main__":
    unittest.main()
