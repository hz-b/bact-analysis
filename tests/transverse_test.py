import unittest
import numpy as np
import xarray as xr
from bact_analysis.transverse import calc, distorted_orbit


class TestKick00_Basis(unittest.TestCase):
    """ """

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


class KickAngleSetup:
    def setUp(self):
        orb = np.sin(np.linspace(0, 2 * np.pi, num=5))
        self.orbit = xr.DataArray(
            data=orb, dims=["pos"], coords=[[f"pos_{p}" for p in range(len(orb))]]
        )

        excitation = [-1, 1]
        self.excitation = xr.DataArray(
            data=excitation,
            dims=["step"],
            coords=[[f"step_{s}" for s in range(len(excitation))]],
        )

        self.measurement = self.excitation * self.orbit
        self.eps = 1e-9


class TestKick10_Angle(KickAngleSetup, unittest.TestCase):
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
        t_pos = "pos_2"
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


class TestKick20_Angle(KickAngleSetup, unittest.TestCase):
    """Improper setup x arrays"""

    def setUp(self):
        super().setUp()
        # Now create measurement with measurement dimensions swapped

        meas = self.measurement
        d = {name: item for name, item in meas.coords.items()}
        n_pos = np.arange(len(meas.coords["pos"]) * 2)
        d["pos"] = n_pos
        t_c = [item for _, item in d.items()]
        m = xr.DataArray(dims=d.keys(), coords=t_c)
        m.data[:, :] = 0.0
        self.measurement = m

    def test_00(self):
        """Mismatch of dimension length will be properly reported

        Currently only testing that an exception is raised. Will trigger
        log message generation
        """
        self.assertRaises(
            ValueError, calc.derive_angle, self.orbit, self.excitation, self.measurement
        )


class TestOrbit(unittest.TestCase):
    """ """

    def setUp(self):
        N = 5
        beta = np.linspace(1, 2, num=N)
        mu = np.linspace(0, (np.pi / 2), num=N)

        dims = ["pos"]
        self.beta_mu = xr.Dataset(
            dict(
                beta=xr.DataArray(data=beta, dims=dims),
                mu=xr.DataArray(data=beta, dims=dims),
            )
        )

        self.eps = 1e-7

    def test30_noDistortion(self):
        dims = ["pos"]
        beta_mu_p = xr.Dataset(
            dict(
                beta=xr.DataArray(data=[0], dims=dims),
                mu=xr.DataArray(data=[0], dims=dims),
            )
        )
        res = distorted_orbit.closed_orbit_distortion(self.beta_mu, beta_mu_p, 0)
        self.assertTrue((np.absolute(res) < self.eps).all())

    def test31_UnitDistortionZero(self):
        dims = ["pos"]

        beta_mu_p = xr.Dataset(
            dict(
                beta=xr.DataArray(data=[0], dims=dims),
                mu=xr.DataArray(data=[0], dims=dims),
            )
        )
        res = distorted_orbit.closed_orbit_distortion(self.beta_mu, beta_mu_p, 0)
        self.assertTrue((np.absolute(res) < self.eps).all())

        tune = 1
        beta_mu_p2 = xr.Dataset(
            dict(
                beta=xr.DataArray(data=[0], dims=dims),
                mu=xr.DataArray(data=[tune], dims=dims),
            )
        )
        beta_mu = xr.concat([self.beta_mu, beta_mu_p2], dim="pos")
        res = distorted_orbit.closed_orbit_distortion(beta_mu, beta_mu_p, 0)
        # Still 0: beta is 0
        self.assertTrue((np.absolute(res) < self.eps).all())


if __name__ == "__main__":
    unittest.main()
