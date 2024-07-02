import unittest
import datetime
import itertools
import numpy as np
import xarray as xr
from bact_analysis.utils import preprocess


def create_xarrays():
    """A xarray for test purposes"""
    vals = ["Q1M1D1R"] * 15 + ["Q1M1D2R"] * 15 + ["Q1M1D1R"] * 15
    currents = list(itertools.chain([0] * 3, [-1] * 3, [0] * 3, [1] * 3, [0] * 3)) * 3

    dt = datetime.timedelta(seconds=0.1)
    time_steps = np.arange(len(vals)) * dt

    arr = xr.DataArray(vals, name="val", dims=["time"], coords=[time_steps])
    arr2 = xr.DataArray(currents, name="cur", dims=["time"], coords=[time_steps])
    return arr, arr2


class TestUniqueSeen(unittest.TestCase):
    """Test xarray functionallity add on

    Test of core functionallity in bact-math-utils
    """

    def test00(self):
        """check enumerate_changed_value: range(10)"""
        steps = list(range(10))
        dt = datetime.timedelta(seconds=0.1)
        time_steps = [r * dt for r in steps]

        arr = xr.DataArray(
            steps,
            dims=["time"],
            coords=[
                time_steps,
            ],
        )
        res = preprocess.enumerate_changed_value(arr)

        dt = arr.coords["time"] - res.coords["time"]
        check = dt.values.astype(int) == 0
        self.assertTrue(check.all())

    def test01(self):
        """Check enumerate_changed_value: typical measurement sequence"""

        arr, arr2 = create_xarrays()
        res = preprocess.enumerate_changed_value_pairs(arr, arr2)

        dt = arr.coords["time"] - res.coords["time"]
        check = dt.values.astype(int) == 0
        self.assertTrue(check.all())

        check_values = (
            (np.arange(15)[:, np.newaxis] * np.ones(3)[np.newaxis, :])
            .ravel()
            .astype(int)
        )
        check = (res.values - check_values) == 0
        self.assertTrue((check).all())

    def test03_reorder_by_vector(self):
        """Test that names are properly added"""

        arr, arr2 = create_xarrays()
        tmp = xr.Dataset(dict(val=arr, cur=arr2))
        grps = tmp.groupby("val")

        a_list = preprocess.reorder_by_groups(
            tmp, grps, reordered_dim="name", dim_sel="time", new_indices_dim="step"
        )

        # Checking that names were set
        self.assertEqual(a_list[0].name, "Q1M1D1R")
        self.assertEqual(a_list[1].name, "Q1M1D2R")

        # Check that the list can be concated
        # typical use case
        res = xr.concat(a_list, dim="name")

        def f_nfinite(vec):
            return np.sum(np.isfinite(vec)).values

        # Find out how many values will expect
        g, g2 = [v for _, v in grps.groups.items()]
        # Vectors are of different size so that gives a few nan's
        values = res.cur.sel(name="Q1M1D1R")
        self.assertEqual(len(g), f_nfinite(values))
        values = res.cur.sel(name="Q1M1D2R")
        self.assertEqual(len(g2), f_nfinite(values))


if __name__ == "__main__":
    unittest.main()
