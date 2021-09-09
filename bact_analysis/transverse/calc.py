from typing import Sequence
import numpy as np
from scipy.linalg import lstsq
import xarray as xr
from bact_math_utils.linear_fit import x_to_cov, cov_to_std


def angle(dist_orb: xr.Dataset, meas_orb: xr.Dataset) -> (xr.Dataset, xr.Dataset):
    """Estimate angle using kick model

    Fits the exictation of the kick and the offset of the quadrupoles
    (thus the ideal orbit does not need to be subtracted beforehand)

    Args:
        dist_orb:          expected distorted orbits: the expected
                           deviation for the different orbit distortions
        measured orbits:   the orbit that were measured.

    The dist_orb needs to be calculated for the different distortions.
    Typically a distorted orbit is multipled with the used magnet excitation.

    Todo:
        find an appropriate name to distinquish between value and error
        result a good one?

    """

    fitres = lstsq(dist_orb, meas_orb)

    # only works if using numpy arrays
    N, p = dist_orb.shape
    cov = x_to_cov(dist_orb.values, fitres[1], N, p)
    std = cov_to_std(cov)

    parameters = xr.DataArray(
        [fitres[0], std],
        dims=["result", "parameter"],
        coords=[["value", "error"], dist_orb.coords["parameter"]],
    )

    return parameters


def for_fitting_reference_orbit(step: Sequence, pos: Sequence) -> xr.DataArray:
    """create array for fitting reference orbit

    Args:
        step: step coordinates
        pos: position coordinates

    Fills array with ones when position and parameter name are the same

    Returns:
        x array of three dimensions: `step`, `pos`, and `parameter`.
        The `parameter` and `pos` labels are the same.

    Todo:
        Review if a sparse array would rather fit the job
    """

    pos_offsets = xr.DataArray(
        0.0, dims=["step", "pos", "parameter"], coords=[step, pos, pos]
    )

    for pos_name in pos:
        pos_offsets.loc[dict(pos=pos_name, parameter=pos_name)] = 1.0

    return pos_offsets


def scale_orbit_distortion(
    orbit: xr.DataArray, excitation: xr.DataArray
) -> xr.DataArray:
    """scale the ideal orbit by the excitation

    Args:
        orbit: distorted orbit due to some kick
        excitation: excitation of the magnet

    Assumes that orbit and excitation are both one dimensional arrays.
    Furthermore assumes that different labels are used.

    Todo:
        can be that it is required for closed orbit calculations too
    """

    (dim_o,) = orbit.dims
    (dim_e,) = excitation.dims

    if dim_o == dim_e:
        txt = "orbit sole dimension name '{}' must not match excitation's sole dimension name '{}'"
        raise AssertionError(txt.format(dim_e, dim_o))

    #: must increase dimensions of array
    result = excitation * orbit
    return result


def derive_angle(
    orbit: xr.DataArray, excitation: xr.DataArray, measurement: xr.DataArray
) -> xr.DataArray:
    """Kicker angle derived from expected orbit, excitation and distortion measurements

    Args:
        orbit:       orbit expected for some excitation (e.g. 10 urad)
        excitation:  different excitations applied to the magnet
        measurement: the measured orbit distortions (containing
                     difference orbit)

    Returns:
        angle and orbit offsets (value and errors)
    """
    sorb = scale_orbit_distortion(orbit, excitation)

    # scale orbit distortion has checkt that dimension names are different
    (dim_o,) = orbit.dims
    (dim_e,) = excitation.dims

    pos = orbit.coords[dim_o]
    step = excitation.coords[dim_e]
    orb_off = for_fitting_reference_orbit(step, pos)

    # perpare array so that offset and angle can be fit at once
    sorb = sorb.expand_dims(parameter=["scaled_angle"])
    A_prep = xr.concat([sorb, orb_off], dim="parameter")

    stack_coords = dict(fit_indep=[dim_e, dim_o])
    A = A_prep.stack(stack_coords)
    meas = measurement.stack(stack_coords)

    # excecute fit
    result = angle(A.T, meas)

    return result


__all__ = [
    "derive_angle",
    "scale_orbit_distortion",
    "for_fitting_reference_orbit",
    "angle",
]
