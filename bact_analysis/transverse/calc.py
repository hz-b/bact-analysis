"""Estimating kick from measurment: fit prepartion and support routines

Main entry point: :func:`derive_angle`
"""
from typing import Sequence
import logging

from scipy.linalg import lstsq
import xarray as xr
import numpy as np
from bact_math_utils.linear_fit import x_to_cov, cov_to_std

logger = logging.getLogger("bact-analysis")


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

        Find the right place to force the measured orbit to a proper dtype
    """

    # enforce that measured orbit is a numpy array
    # todo
    meas_orb = np.array(meas_orb, dtype=float)
    fitres = lstsq(dist_orb, meas_orb)
    _, residues, rank, _ = fitres
    N, p = dist_orb.shape

    if rank != p:
        raise AssertionError(f"Fit with {p} parameters returned a rank of {rank}")

    # only works if using numpy arrays
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
        txt = (
            "orbit sole dimension name '{}' must not match excitation's"
            " sole dimension name '{}'"
        )
        raise AssertionError(txt.format(dim_e, dim_o))

    #: must increase dimensions of array
    result = excitation * orbit
    return result


def derive_angle(
    orbit: xr.DataArray,
    excitation: xr.DataArray,
    measurement: xr.DataArray,
    *,
    weights: xr.DataArray = None,
) -> xr.DataArray:
    """Kicker angle derived from expected orbit, excitation and distortion measurements

    Args:
        orbit:       orbit expected for some excitation (e.g. 10 urad)
        excitation:  different excitations applied to the magnet
        measurement: the measured orbit distortions (containing
                     difference orbit)
        weights (default =None): weights of the measurements

    Returns:
        angle and orbit offsets (value and errors)
    """
    if weights is None:
        sqw = None
    else:
        sqw = np.sqrt(weights)

    # scale orbit distortion has checks that dimension names are different
    (dim_o,) = orbit.dims
    (dim_e,) = excitation.dims

    sorb = scale_orbit_distortion(orbit, excitation)

    pos = orbit.coords[dim_o]
    step = excitation.coords[dim_e]
    orb_off = for_fitting_reference_orbit(step, pos)

    # perpare array so that offset and angle can be fit at once
    sorb = sorb.expand_dims(parameter=["scaled_angle"])
    # logger.debug("droping dimension name")
    # sorb = sorb.drop(dims=['name'])
    dim_parameter = "parameter"
    try:
        A_prep = xr.concat([sorb, orb_off], dim=dim_parameter)
    except Exception as exc:
        logger.error('Failed to concat along dim "%s": reason %s', dim_parameter, exc)
        logger.error("sorb dims  %s coords %s", sorb.dims, sorb.coords)
        logger.error("orb_off %s", orb_off)
        raise exc


    # Scale independents matrix with weights
    # If the orb_off is not scaled  the result will be "interesting"
    if sqw is not None:
        try:
            A_prep = A_prep * sqw
        except Exception as exc:
            txt = (
                f"{__name__}.derive_angle: Failed to apply weights: {exc}"
                f" sorb.shape : {sorb.shape} weight dims {sqw.shape}"
                f"types: sorb : {type(sorb)} weight dims {type(sqw)}"
            )
            logger.error(txt)
            logger.error(f" sorb.dims : {sorb.dims} weight dims {sqw.dims}")
            raise exc

    stack_coords = dict(fit_indep=[dim_e, dim_o])
    A = A_prep.stack(stack_coords)

    # scale measured data "b" column
    if sqw is not None:
        try:
            measurement = measurement * sqw
        except Exception as exc:
            txt = (
                f"{__name__}.derive_angle: Failed to apply weights: {exc}"
                f" measurement.dims : {measurement.dims} weight dims {sqw.dims}"
            )
            logger.error(txt)
            raise exc

    try:
        meas = measurement.stack(stack_coords)
    except Exception as exc:
        msg = (
            f"Request for stacking coordinates {stack_coords},"
            f" but measurement had only coords {measurement.coords}"
        )
        logger.error(msg)
        raise exc
    # excecute fit
    try:
        result = angle(A.T, meas)
    except ValueError as exc:
        txt = (
            f" {__name__}:derive_angle"
            f" A dims {A.dims} with shape {A.shape}"
            f" b dims {meas.dims} with shape {meas.shape}"
            f" A_prep dims {A_prep.dims} with shape {A_prep.shape}"
            f" measurement dims {measurement.dims} with shape {measurement.shape}."
            f" The dimensions {dim_o} and {dim_e} must match in A_prep and measurement."
            f" The dimension 'paramter' should match length of {dim_o} + 1"
        )
        logger.error(txt)
        raise exc

    return result


__all__ = [
    "derive_angle",
    "scale_orbit_distortion",
    "for_fitting_reference_orbit",
    "angle",
]
