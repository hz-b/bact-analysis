"""Utilities for closed orbit calculation

Inputs expected as :mod:`xarrays`.

Uses implementation of :mod:`bact-math-utils`
"""
import logging
import xarray as xr
from bact_math_utils import distorted_orbit as mdo

logger = logging.getLogger("bact-analysis")


def closed_orbit_distortion(
    beta_mu: xr.Dataset,
    beta_mu_p: xr.Dataset,
    theta: float,
    *,
    tune_pos: str = None,
    scale_tune: float = 1.0,
    scale_phase_advance: float = 1.0
) -> xr.DataArray:
    """Calculate orbit distortion created by one kicker

    Args:
        beta_mu:   betatron function and phase propagation along
                   the ring
        beta_mu_p: betatron function and phase propagation at the
                   distortion point
        theta:     angle created at distortion point


    Returns:
        closed orbit distortion

    Warning:
        Assumes that the working point is contained within beta_mu
    """

    (dim,) = beta_mu.dims
    beta = beta_mu.beta
    mu = beta_mu.mu

    beta_i = beta_mu_p.beta.values
    mu_i = beta_mu_p.mu.values

    if tune_pos is None:
        tune = mu.isel({dim: -1})
    else:
        tune = mu.sel({dim: tune_pos})
    tune = tune.values * scale_tune

    mu = mu * scale_phase_advance
    mu_i = mu_i * scale_phase_advance

    fmt = (
        "%s.closed_orbit_distortion using tune %s last phase advance %s,"
        " phase adavance at kick %s"
    )
    logger.debug(fmt, __name__, tune, mu.isel(pos=-1).values, mu_i)

    ra = mdo.closed_orbit_distortion(
        beta.values, mu.values, tune=tune, beta_i=beta_i, theta_i=theta, mu_i=mu_i
    )

    res = xr.DataArray(
        name="distorted_orbit",
        data=ra,
        dims=[dim],
        coords=[beta_mu.coords[dim]],
        attrs=dict(theta=theta, scale_tune=scale_tune, scale_phase_advance=scale_phase_advance),
    )
    res

    return res


__all__ = ["closed_orbit_distortion"]
