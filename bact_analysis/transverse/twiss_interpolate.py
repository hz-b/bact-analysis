"""Interpolate Twiss functions

Beam dynamics programs typically use the element name to indicate
the start or the end of an element. When calculating the distorted
orbit the effect of the integral :math:`\int \beta(s) ds` is
approximated by an rectangle using :math:`\beta(s=l/2)` with l the
have length of the element.

Convienience functions provided here
"""
import xarray as xr
import numpy as np
from typing import Sequence
import logging

logger = logging.getLogger("bact-analysis")


def data_for_elements(
    input: xr.Dataset,
    names: Sequence[str],
    *,
    coordinate_name: str = "pos",
    name_dim: str = "name",
    element_dim: str = "element",
) -> xr.Dataset:
    """
    Args:
        input:           dataset containing the betatron functions.
                         It must be sorted by element position. It
                         is assumed that the name indicates the
                         start of the element
        names:           position names to be used
        coordinate_name: dimension name that contains the position names
        name_dim:        dimension name that will contain position names
                         in returned array
        element_dim:     dimension name that will constain start and end

    Returns:
         dataset with start and end of element

    Todo:
        Check if test of already existing dimension name should be
        only made if adding element fails.
        Review coordinate name variable.
        Check if xarray has a more straightforward use case
    """
    pos = input.coords.indexes[coordinate_name]

    # Here the index has to be fou
    def t_index(name):
        idx = pos.get_loc(name)
        if idx < 1:
            txt = (
                f"Index {idx} for element {name} < 1."
                " Element expected to extend from (index - 1)  to index"
            )
            raise AssertionError(txt)

        istart = idx - 1
        iend = idx
        assert istart >= 0
        return (istart, iend)

    indices = {name: t_index(name) for name in names}

    # For start and end of each element
    dims = list(input.dims)
    for ndim in [name_dim, element_dim]:
        if element_dim in dims:
            txt = (
                f"I need intermediate dimension {element_dim}."
                f" But it is already contained in dimensions {dims}"
            )
            raise AssertionError(txt)

    # rearrange your data
    d_rename_dim = {coordinate_name: element_dim}
    d_nelem_dim = {element_dim: ["start", "end"]}

    def start_and_end(name, indices):
        start, end = indices
        data = (
            input.isel({coordinate_name: [start, end]})
            .rename_dims(d_rename_dim)
            .assign_coords(d_nelem_dim)
            .expand_dims({name_dim: [name]})
            .reset_coords(drop=True)
        )
        return data

    data = [start_and_end(name, indices) for name, indices in indices.items()]
    try:
        res = xr.concat(data, dim=name_dim)
    except:
        fmt = "%s.data_for_elements processing data:\n%s\n on dimension %s"
        logger.error(fmt, __name__, data[0], name_dim)
        raise
    # res = res.rename_dims(d_rename_dim).assign_coords(
    #    {element_name_dim: [name for name, _ in indices.items()]}
    # )
    return res


def interpolate_twiss(
    input: xr.Dataset,
    names: Sequence[str],
    *,
    coordinate_name: str = "pos",
    name_dim: str = "name",
    element_dim: str = "element",
    **kwargs,
) -> xr.Dataset:
    """ """
    data = data_for_elements(
        input,
        names,
        coordinate_name=coordinate_name,
        element_dim=element_dim,
        name_dim=name_dim,
        **kwargs,
    )
    # Now processing is pretty straight forward
    res = data.mean(dim=element_dim)
    return res
