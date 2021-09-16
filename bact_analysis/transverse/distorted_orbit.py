"""Utilities for closed orbit calculation

Todo:
    Use consistent variables: tune in mu or phi?
    Tune: call it tune or working point?
"""

import numpy as np


def closed_orbit_kick_unscaled(mu: np.array, *, tune: float, mu_i: float) -> np.ndarray:
    """Calculated closed orbit kick (not scaled by beta)

    Args:
        mu:   phase propagation
        tune: working point
        mu_i: phase propagation of kicker

    Returns: unscaled kick

    .. math::

        \\cos{\\left(
            \\pi Q -
                  \\left|{\\mu_i - \\mu(s)}
                 \\right|
         \\right)}

    """
    mu = np.asarray(mu)

    qp = tune * np.pi
    dmu = mu_i - mu
    dmua = np.absolute(dmu)

    r = np.cos(qp - dmua)
    return r


def closed_orbit_kick(
    mu: np.ndarray, *, tune: float, beta_i: float, theta_i: float, mu_i: float
) -> np.ndarray:
    """Calculated closed orbit kick  (scaled by beta and theta)

    Args:
        mu:     phase propagation
        tune:   working point
        mu_i:   phase propagation of kicker
        beta_i: beta at the kicker
        theta_: kicker aangle

    Returns: kick scaled by sqrt of beta and angle

    .. math::
        \\sqrt{\\beta_i}  \\vartheta_i \\cdot closed\\_orbit\\_kick\\_unscaled
    """
    cou = closed_orbit_kick_unscaled(mu, tune=tune, mu_i=mu_i)
    scale = np.sqrt(beta_i) * theta_i
    r = scale * cou
    return r


def closed_orbit_distortion(
    beta: np.ndarray,
    mu: np.ndarray,
    *,
    tune: float,
    beta_i: float,
    theta_i: float,
    mu_i: float
) -> np.ndarray:
    r"""Calculate orbit distortion created by one kicker

    Args:
        beta:   betatron function along the ring
        mu:     phase propagation
        tune:   working point
        mu_i:   phase propagation of kicker
        beta_i: beta at the kicker
        theta_i: kicker angle

    Returns:
        closed orbit distortion

    .. math::

        \frac{\sqrt{\beta(s)}}{2 \sin{\left(\pi Q\right)}}\cdot closed\_orbit\_kick


    full equation

    .. math::

        \frac{\sqrt{\beta(s)}}{2 \sin{\left(\pi Q\right)}}
        \cdot  \vartheta_i \sqrt{\beta_i}
        \cos{\left(
            \pi Q -
                  \left|\mu_i - \mu(s) \right|
         \right)}


    """
    devisor = 2.0 * np.sin(tune * np.pi)
    scale = 1.0 / devisor

    beta = np.asarray(beta)
    sq_beta = np.sqrt(beta)

    cok = closed_orbit_kick(mu, tune=tune, beta_i=beta_i, theta_i=theta_i, mu_i=mu_i)

    r = scale * sq_beta
    r *= cok
    return r


__all__ = ["closed_orbit_distortion", "closed_orbit_kick_unscaled", "closed_orbit_kick"]
