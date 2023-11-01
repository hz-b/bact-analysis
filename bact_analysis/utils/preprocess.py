""" Preprocess measurement data
"""
from typing import Sequence
import logging
import copy
import numpy as np
import xarray as xr
import xarray.core.groupby
from bact_math_utils import misc as bmu

logger = logging.getLogger("bact_analysis")


def enumerate_changed_value(vec: xr.DataArray) -> xr.DataArray:

    (dim,) = vec.dims
    data = bmu.enumerate_changed_value(vec.values)

    changed_values = xr.DataArray(data, dims=[dim], coords=[vec.coords[dim]])
    return changed_values


def enumerate_changed_value_pairs(
    vec1: xr.DataArray, vec2: xr.DataArray
) -> xr.DataArray:

    (dim,) = vec1.dims
    (dim_check,) = vec2.dims

    assert dim == dim_check

    data = bmu.enumerate_changed_value_pairs(vec1.values, vec2.values)
    changed_values = xr.DataArray(data, dims=[dim], coords=[vec1.coords[dim]])
    return changed_values


def check_variable_replaceable_dim(var, expected_length: int, dim_to_ignore=None):
    """Check if a not ignored dimension of the variable could be replaced

    Todo:
        Review naming of the function
    """
    assert dim_to_ignore is not None
    dims = list(var.dims)
    dims_rm = dims[:]
    dims_rm.remove(dim_to_ignore)

    try:
        (dim,) = dims_rm
    except Exception as exc:
        name = var.name
        logger.error(
            f"Handling variable {name}: dims {dims}"
            f" dims_rm {dims_rm} dim_to_ignore {dim_to_ignore}"
        )
        raise exc

    coord = var.coords[dim]

    lc = len(coord)
    if expected_length is not None and lc != expected_length:
        txt = (
            f"variable {var.name} sole not ignored dimension {dim}"
            f" length {lc} does not match expected length {expected_length}"
        )
        raise AssertionError(txt)

    return dim


def replaceable_dims(
    dataset,
    variable_names: Sequence,
    prefix="",
    expected_length=None,
    dim_to_ignore="time",
) -> list:
    """

    Args:
        dataset:        compatible to :class:`xarray.Dataset`
        prefix:         prefix of the variable
        dim_to_ignore:  variables typically have more than one
                        dimension. Typically the time variable
                        is not changed here

    Todo:
        Consider making it a lazy evaluated function
    """

    var_names = [prefix + name for name in variable_names]

    dims = [
        check_variable_replaceable_dim(
            dataset[var_name], expected_length, dim_to_ignore=dim_to_ignore
        )
        for var_name in var_names
    ]
    return dims


def reindex_slice(
    xs: xr.Dataset, *, dim_sel: str, indices: Sequence[int], new_indices_dim: str
) -> xr.Dataset:
    """Add coordinate at new_dim_name with consecutive index added

    xs:               a :class:`xarray.Dataset` like object
    dim_sel:          dimension where to use the indices
    indices:          indices to select at dimension dim_sel
    new_indices_dim:  where to add the new index (range of len(indices))
    """

    # select the relevant part
    sel = xs.isel({dim_sel: indices})

    # create a new index with steps
    new_indices = np.arange(len(indices))

    res = sel.rename({dim_sel: new_indices_dim}).assign_coords(
        {new_indices_dim: new_indices}
    )
    return res


def reorder_by_groups(
    xs: xr.Dataset, groups: xr.core.groupby.DatasetGroupBy, *, reordered_dim, **kwargs
):
    """reorder data by group.

    See :func:`reindex_slice` for requried keyword arguments
    """

    def process(*args, name=None, **kwargs):
        assert name is not None

        res = reindex_slice(*args, **kwargs)
        res = res.expand_dims({reordered_dim: [name]})
        return res

    result = [
        process(xs, indices=indices, name=name, **kwargs)
        for name, indices in groups.groups.items()
    ]

    return result


def rename_doublicates(names: list) -> [dict, list]:
    """Later names get a suffix added"""

    # How to do that in dict comprehension without calling names.count twice
    d = {}
    for n in names:
        cnt = names.count(n)
        if cnt > 1:
            d[n] = cnt

    new_names = copy.copy(names)

    for name, n_cnt in d.items():
        # Only handle ones which occur twice
        assert n_cnt == 2
        idx = new_names.index(name)
        new_names[idx] = name + "_d"

    return d, new_names


def replace_names(names: list, replacement_names: dict) -> list:
    """Replace  names with replacement names

    Assumes that entries names are unique. If not behaviour is
    undefined.

    Used for preparing model for bba calculations
    """

    repl = list(replacement_names.keys())

    n_names = names.copy()
    for t_name, replacement in replacement_names.items():
        idx = n_names.index(t_name)
        n_names[idx] = replacement

    return n_names


__all__ = [
    "rename_doublicates",
    "reorder_by_groups",
    "enumerate_changed_value",
    "enumerate_changed_value_pairs",
    "reindex_slice",
    "replace_names",
]
