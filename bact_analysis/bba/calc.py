"""
"""
import dataclasses


def angle_to_offset(tf: float, length: float, polarity: int, alpha: float) -> float:
    r"""Derive offset from measured specific kick angle

    Args:
        tf:       central quadrupole strength K1  per excitation
        length:   magnet length
        polarity: polarity of the circuit
        alpha:    angle per unit excitation


    A (quadrupole) offset :math:`\Delta x_{quad}` gives an kick
    angle of

    .. math::

        \frac{\Delta \vartheta}{\Delta I} =
                \frac{\Delta K_1}{\Delta I}\, L \,\Delta x_{quad}

    Here the specific kick angle :math:`\alpha` and the specific
    exitation :math:`t_f` are used.

    .. math::
        \alpha = \frac{\Delta \vartheta}{\Delta I} \qquad
         t_f = \frac{\Delta K_1}{\Delta I}

    Thus one obtains

    .. math::
        \Delta x_{quad} = \frac{\alpha}{L \, t_f}

    """

    devisor = tf * polarity * length
    offset = alpha / devisor
    return offset


@dataclasses.dataclass
class MagnetInfo:
    length: float
    tf: float
    polarity: int

    def angle_to_offset(self, angle):
        return angle_to_offset(self.tf, self.length, self.polarity, angle)


__all__ = ["MagnetInfo", "angle_to_offset"]
