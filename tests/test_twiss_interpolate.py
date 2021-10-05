import unittest
from bact_analysis.transverse import twiss_interpolate
import xarray as xr


class TestDataOne:
    def createData(self):
        self.name = "q4m2d1r"
        pos = ["some_name", self.name]

        beta = xr.DataArray(data=[0, 2], dims=["pos"], coords=[pos])
        nu = xr.DataArray(data=[3, 5], dims=["pos"], coords=[pos])
        self.data = xr.Dataset(dict(beta=beta, nu=nu))

    def setUp(self):
        self.createData()


class TestDataTwo:
    def createData(self):

        self.name = "q4m1d1r"
        self.name2 = "q4m2d1r"
        pos = ["some_name", self.name, "some_element", self.name2]

        beta = xr.DataArray(data=[0, 2, 7, 11], dims=["pos"], coords=[pos])
        nu = xr.DataArray(data=[3, 5, 17, 19], dims=["pos"], coords=[pos])
        self.data = xr.Dataset(dict(beta=beta, nu=nu))

    def setUp(self):
        self.createData()


class _TwissInterpolateErrorPos(unittest.TestCase):
    def setUp(self):
        super().createData()
        pos = list(self.data.coords["pos"].values)
        npos = [pos[1], pos[0]] + pos[2:]
        data = self.data.copy().assign_coords(pos=npos)
        self.data = data

    def test00_checkIndex(self):
        """Test if found that element at wrong position"""

        # Check that swap worked
        assert self.data.coords["pos"][0] == self.name
        self.assertRaises(
            AssertionError, twiss_interpolate.data_for_elements, self.data, [self.name]
        )


class _TwissInterpolateErrorIntermediateDim(unittest.TestCase):
    intermediate_dim = "element"

    def setUp(self):
        super().createData()

    def test00_checkExtraDim(self):
        """Intermediate data already used as dimension"""
        data = self.data.expand_dims({self.intermediate_dim: ["TroubleMaker"]})
        self.assertRaises(
            AssertionError, twiss_interpolate.data_for_elements, data, [self.name]
        )

    def test01_checkExtraDim(self):
        """Intermediate data already used as position dimension"""
        data = self.data.rename_dims({"pos": self.intermediate_dim}).assign_coords(
            {self.intermediate_dim: self.data.coords["pos"].values}
        )
        self.assertRaises(
            AssertionError,
            twiss_interpolate.data_for_elements,
            data,
            [self.name],
            coordinate_name=self.intermediate_dim,
        )


class _TwissInterpolate(unittest.TestCase):
    """ """

    def setUp(self):
        super().createData()

    def test00_checkDataForElements(self):
        """Standard usage"""
        data = twiss_interpolate.data_for_elements(self.data, [self.name])
        (t_name,) = data.coords["name"]
        self.assertEqual(t_name, self.name)

        start, end = data.coords["element"]
        self.assertEqual(start, "start")
        self.assertEqual(end, "end")


class TwissInterpolateErrorPosOne(_TwissInterpolateErrorPos, TestDataOne):
    pass


class TwissInterpolaterErrorPosTwo(_TwissInterpolateErrorPos, TestDataTwo):
    pass


class TwissInterpolateErrorIntermediateDimOne(
    _TwissInterpolateErrorIntermediateDim, TestDataOne
):
    pass


class TwissInterpolaterErrorIntermediateDimTwo(
    _TwissInterpolateErrorIntermediateDim, TestDataTwo
):
    pass


class TwissInterpolateOne(_TwissInterpolate, TestDataOne):
    pass


class TwissInterpolaterTwo(_TwissInterpolate, TestDataTwo):
    def test10_checkDataForElements(self):
        """More than one correctly returned?"""
        try:
            data = twiss_interpolate.data_for_elements(
                self.data, [self.name2, self.name]
            )
        except:
            print("Used test data\n", self.data)
            raise

        t_name2, t_name = data.coords["name"]
        self.assertEqual(t_name, self.name)
        self.assertEqual(t_name2, self.name2)

        beta = data.beta.sel(name=self.name).values
        diff = data - self.data
        self.assertEqual(diff.max(), 0)

    def test11_checkDataForElements(self):
        """More than one correctly returned? Reversed order

        Perhaps a bit parnoid ....
        """
        data = twiss_interpolate.data_for_elements(self.data, [self.name, self.name2])
        t_name, t_name2 = data.coords["name"]  # .values
        self.assertEqual(t_name, self.name)
        self.assertEqual(t_name2, self.name2)

    def checkReturn(self, data):
        beta = data.beta.sel(name=self.name)
        self.assertEqual(beta, 1)
        nu = data.nu.sel(name=self.name)
        self.assertEqual(nu, 4)

        beta = data.beta.sel(name=self.name2)
        self.assertEqual(beta, 9)
        nu = data.nu.sel(name=self.name2)
        self.assertEqual(nu, 18)

    def test20_checkDataForElements(self):
        data = twiss_interpolate.interpolate_twiss(self.data, [self.name, self.name2])
        self.checkReturn(data)

    def test21_checkDataForElements(self):
        data = twiss_interpolate.interpolate_twiss(self.data, [self.name2, self.name])
        self.checkReturn(data)


del _TwissInterpolateErrorPos, _TwissInterpolateErrorIntermediateDim, _TwissInterpolate

if __name__ == "__main__":
    unittest.main()
