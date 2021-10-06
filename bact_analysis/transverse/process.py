"""

Todo:
   better naming of module!
"""

from .calc import derive_angle
from .distorted_orbit import closed_orbit_distortion
import xarray as xr
import numpy as np
from typing import Sequence
import logging
from dataclasses import dataclass

logger = logging.getLogger("bact-analysis")


@dataclass
class Result:
    """Typically only required for storing intermediate results"""

    orbit: xr.DataArray
    orbit_at_bpm: xr.DataArray
    result: xr.DataArray


def process_magnet_plane(
    selected_model: xr.Dataset,
    selected_model_for_magnet: xr.Dataset,
    excitation: xr.DataArray,
    offset: xr.DataArray,
    *,
    weights: xr.DataArray = None,
    bpm_names: Sequence,
    theta: float,
    scale_tune: float = 1,
    scale_phase_advance: float = 2 * np.pi,
) -> Result:

    # orbit is supposed to contain theta, scale_tune and scale_phase_advance in its metadata
    orbit = closed_orbit_distortion(
        selected_model,
        selected_model_for_magnet,
        theta=theta,
        scale_tune=1,
        scale_phase_advance=2 * np.pi,
    )
    orbit_at_bpm = orbit.sel(pos=bpm_names)

    try:
        res = derive_angle(
            orbit=orbit_at_bpm,
            excitation=excitation,
            measurement=offset,
            weights=weights,
        )
    except ValueError as exc:
        txt = (
            f" {__name__}:process_magnet_plane:"
            f" orbit  dimensions {orbit.dims} shape {orbit.shape}"
            f" excitation dimensions {excitation.dims} shape {excitation.shape}"
            f" measurement dimensions {offset.dims} shape {offset.shape}"
        )
        logger.error(txt)
        raise exc

    r = Result(orbit=orbit, orbit_at_bpm=orbit_at_bpm, result=res)
    return r


def process_magnet(
    selected_model: xr.Dataset,
    selected_model_for_magnet: xr.Dataset,
    measurement: xr.Dataset,
    *,
    bpm_names,
    use_weights=False,
    **kwargs,
) -> xr.Dataset:

    if use_weights:
        x_weights = 1.0 / measurement.x_rms
        y_weights = 1.0 / measurement.y_rms
    else:
        x_weights = None
        y_weights = None

    x_res = process_magnet_plane(
        selected_model.sel(plane="x"),
        selected_model_for_magnet.sel(plane="x"),
        measurement.excitation,
        measurement.x_pos,
        weights=x_weights,
        bpm_names=bpm_names,
        **kwargs,
    )
    y_res = process_magnet_plane(
        selected_model.sel(plane="y"),
        selected_model_for_magnet.sel(plane="y"),
        measurement.excitation,
        measurement.y_pos,
        weights=y_weights,
        bpm_names=bpm_names,
        **kwargs,
    )

    # Rearrange data for planes
    def concat(x, y):
        tmp = [
            x.expand_dims(plane=["x"]),
            y.expand_dims(plane=["y"]),
        ]
        return xr.concat(tmp, dim="plane")

    orbit = concat(x_res.orbit, y_res.orbit)
    # orbit_at_bpm = concat(x_res.orbit_at_bpm.rename(pos="pos_bpm"), y_res.orbit_at_bpm)
    # results of the fits
    result = concat(x_res.result, y_res.result)
    r = xr.Dataset(
        dict(orbit=orbit, fit_params=result), attrs=dict(bpm_names=bpm_names)
    )
    return r


def process_all_gen(
    selected_model: xr.Dataset,
    measurement: xr.Dataset,
    magnet_names: Sequence,
    **kwargs,
) -> (str, xr.Dataset):
    def f(name):
        mod4mag = selected_model.sel(pos=name.lower())
        meas4mag = measurement.sel(name=name)
        return process_magnet(selected_model, mod4mag, meas4mag, **kwargs)

    for name in magnet_names:
        yield name, f(name)


def process_all(
    selected_model: xr.Dataset,
    measurement: xr.Dataset,
    magnet_names: Sequence,
    **kwargs,
) -> dict:

    d = {
        name: item
        for name, item in process_all_gen(
            selected_model, measurement, magnet_names, **kwargs
        )
    }
    return d


def combine_all(input: dict) -> xr.Dataset:
    names = list(input.keys())

    # check that subarray exists at least for the first one
    name = names[0]
    ds = input[name]
    # A bit too frequently used
    del name

    ds.orbit
    ds.fit_params
    bpm_names = ds.attrs["bpm_names"]
    bpm_n_prep = set(bpm_names)

    def process(ds, name):
        check = set(ds.attrs["bpm_names"])
        assert len(list(bpm_n_prep.difference(check))) == 0
        assert len(list(check.difference(bpm_n_prep))) == 0
        # Efficient check that all bpm_names are the same?
        return ds.expand_dims(name=[name])

    l = [process(item, t_name) for t_name, item in input.items()]
    ds = xr.concat(l, dim="name")
    return ds
