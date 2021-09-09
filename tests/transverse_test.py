import unittest
import numpy as np
import xarray as xr
from bact_analysis.transverse import calc, distorted_orbit


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

        self.assertTrue((np.absolute(res) < self.eps).all())

    def test21_DeriveAnglesNoOffset(self):
        """corrrect angle when no offsets"""
        res = calc.derive_angle(self.orbit, self.excitation, self.measurement)

        d = dict(parameter="scaled_angle", result="value")
        self.assertAlmostEqual(res.loc[d], 1)
        res.loc[d] = 0

        # all others zero
        self.assertTrue((np.absolute(res) < self.eps).all())

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
        self.assertTrue((np.absolute(res) < self.eps).all())


class TestOrbit(unittest.TestCase):
    """ """

    def setUp(self):
        N = 5
        self.beta = np.linspace(1, 2, num=N)
        self.mu = np.linspace(0, (np.pi / 2), num=N)
        self.eps = 1e-7

    def test10_UnitUnscaledKick(self):
        """negative side so to say ...."""
        tune = 1

        f = distorted_orbit.closed_orbit_kick_unscaled
        res = f(self.mu, tune=tune, mu_i=0)
        test = np.cos(tune * np.pi - self.mu)
        self.assertTrue((np.absolute(res - test) < self.eps).all())

    def test11_UnitUnscaledKick(self):
        """positive side so to say ...."""
        tune = 1
        mu_i = tune * np.pi

        f = distorted_orbit.closed_orbit_kick_unscaled
        res = f(self.mu, tune=tune, mu_i=mu_i)
        test = np.cos(-self.mu)
        self.assertTrue((np.absolute(res - test) < self.eps).all())

    def test20_UnitScaledKick(self):
        tune = 1

        f = distorted_orbit.closed_orbit_kick
        res = f(self.mu, tune=tune, mu_i=0, theta_i=1, beta_i=1)
        test = np.cos(tune * np.pi - self.mu)
        self.assertTrue((np.absolute(res - test) < self.eps).all())

    def test20_UnitScaledKick2(self):
        tune = 1

        f = distorted_orbit.closed_orbit_kick
        res = f(self.mu, tune=tune, mu_i=0, theta_i=2, beta_i=4)
        test = np.cos(tune * np.pi - self.mu)
        # One two for the angle and one for the squared beta_i
        test *= 2 * 2
        self.assertTrue((np.absolute(res - test) < self.eps).all())

    def test30_noDistortion(self):
        res = distorted_orbit.closed_orbit_distortion(
            self.beta, self.mu, tune=self.mu.max() * 2, beta_i=0, theta_i=0, mu_i=0
        )
        self.assertTrue((np.absolute(res) < self.eps).all())

    def test31_UnitDistortionZero(self):
        tune = 1

        res = distorted_orbit.closed_orbit_distortion(
            self.beta, self.mu, tune=tune, beta_i=0, theta_i=1, mu_i=0
        )
        # Still 0: beta is 0
        self.assertTrue((np.absolute(res) < self.eps).all())

    def test32_UnitDistortion(self):
        tune = 1 + 1 / 2

        test = np.cos(self.mu)

        beta = np.ones(self.beta.shape)
        res = distorted_orbit.closed_orbit_distortion(
            beta, self.mu, tune=tune, beta_i=1, theta_i=1, mu_i=tune * np.pi
        )
        self.assertTrue((np.absolute(res + test / 2) < self.eps).all())

        beta_sq = np.arange(1, 6)
        beta = beta_sq ** 2
        res = distorted_orbit.closed_orbit_distortion(
            beta, self.mu, tune=tune, beta_i=1, theta_i=1, mu_i=tune * np.pi
        )
        self.assertTrue((np.absolute(res + beta_sq * test / 2) < self.eps).all())


if __name__ == "__main__":
    unittest.main()
