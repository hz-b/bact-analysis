import unittest
import numpy as np
import xarray as xr
from bact_analysis.transverse import calc


class TestKick00_Basis(unittest.TestCase):
    def test00_Offset(self):
        """Creation of offset matrix"""

        pos = ["a", "b", "c"]
        step = [1, 2, 3]

        arr = calc.for_fitting_reference_orbit(step, pos)

        self.assertEqual(len(pos), len(arr.coords["pos"]))
        self.assertEqual(len(pos), len(arr.coords["parameter"]))
        self.assertEqual(len(step), len(arr.coords["step"]))

        # Check that all matching positions are 1
        for p in pos:
            d = dict(pos=p, parameter=p)
            tmp = arr.loc[d]
            for v in tmp:
                self.assertAlmostEqual(v, 1)

        # Check that everything else is 0: set the ones that
        # Where checked to be ones to 0
        for p in pos:
            d = dict(pos=p, parameter=p)
            arr.loc[d] = 0

        # Everything must be 0 now
        self.assertAlmostEqual(arr.sum(), 0)

    def test10_Scale(self):
        """Test that scaling produces corrrect results"""

        orbit = np.sin(np.linspace(0, np.pi * 2, num=5))
        excitation = [1, -2, 3]

        orbit = xr.DataArray(orbit, dims=["pos"])
        excitation = xr.DataArray(excitation, dims=["step"])

        res = calc.scale_orbit_distortion(orbit, excitation)

        for scale, line in zip(excitation, res):
            line = line.values
            self.assertAlmostEqual(line[0], 0)
            self.assertAlmostEqual(line[2], 0)
            self.assertAlmostEqual(line[4], 0)
            self.assertAlmostEqual(line[1], scale)
            self.assertAlmostEqual(line[3], -scale)

    def test11_ScaleCheckDimNames(self):
        """Check that it is detected if both have the same name"""
        orbit = np.sin(np.linspace(0, np.pi * 2, num=5))
        excitation = np.linspace(-1, 1, num=3)

        orbit = xr.DataArray(orbit, dims=["pos"])
        excitation = xr.DataArray(excitation, dims=["pos"])

        self.assertRaises(
            AssertionError, calc.scale_orbit_distortion, orbit, excitation
        )


class TestKick10_Angle(unittest.TestCase):
    def setUp(self):
        orb = np.sin(np.linspace(0, 2 * np.pi, num=5))
        self.orbit = xr.DataArray(data=orb, dims=["pos"])

        self.excitation = xr.DataArray(data=[-1, 1], dims=["step"])

        self.measurement = self.excitation * self.orbit
        self.eps = 1e-9

    def test20_DeriveAngleZeros(self):
        """Check that fit works when all data are zero"""

        res = calc.derive_angle(self.orbit, self.excitation, 0 * self.measurement)
        name_d1, name_d2 = res.dims
        self.assertEqual(name_d1, "result")
        self.assertEqual(name_d2, "parameter")

        self.assertTrue((res < self.eps).all())

    def test21_DeriveAnglesNoOffset(self):
        """corrrect angle when no offsets"""
        res = calc.derive_angle(self.orbit, self.excitation, self.measurement)

        d = dict(parameter="scaled_angle", result="value")
        self.assertAlmostEqual(res.loc[d], 1)
        res.loc[d] = 0

        # all others zero
        self.assertTrue((res < self.eps).all())

    def test21_DeriveAnglesSingleOffset(self):
        """See if offset if found if a single value is deviated"""
        t_pos = 2
        val = 2
        meas = self.measurement.copy()
        meas.loc[dict(pos=t_pos)] = val
        res = calc.derive_angle(self.orbit, self.excitation, meas)

        d = dict(parameter="scaled_angle", result="value")
        self.assertAlmostEqual(res.loc[d], 1)
        res.loc[d] = 0

        d = dict(parameter=t_pos, result="value")
        self.assertAlmostEqual(res.loc[d].values, val)
        res.loc[d] = 0

        # all others zero
        self.assertTrue((res < self.eps).all())


if __name__ == "__main__":
    unittest.main()
