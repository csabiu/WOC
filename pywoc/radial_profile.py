import numpy as np

__all__ = ['radial_profile']

def radial_profile(data, mask, center, rmin, rmax, width, method='median'):
    """Compute the radial profile of *data* around *center*.

    Parameters
    ----------
    data : ndarray
        2-D or 3-D array whose radial statistics are computed.
    mask : ndarray
        Boolean mask array with the same shape as ``data`` where valid pixels
        are marked with ``1``.
    center : tuple
        ``(x, y)`` coordinates for 2D or ``(x, y, z)`` coordinates for 3D
        of the profile centre.
    rmin, rmax : float
        Minimum and maximum radius to consider.
    width : float
        Bin width.
    method : {"median", "mean"}, optional
        Statistic to compute for each radial bin.

    Returns
    -------
    r : ndarray
        Radii at the centre of each bin.
    array : ndarray
        Computed statistic for each radial bin.
    """

    # Determine radius of every pixel/voxel relative to the centre
    ndim = data.ndim
    if ndim == 2:
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    elif ndim == 3:
        z, y, x = np.indices(data.shape)
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    else:
        raise ValueError(f"data must be 2D or 3D, got {ndim}D")

    sim = data.ravel()
    sr = r.ravel()
    msk = mask.ravel().copy()

    # Exclude invalid values
    msk[(sim == -np.inf) | (sim == np.inf)] = 0
    valid = (msk == 1) & np.isfinite(sim)

    sim = sim[valid]
    sr = sr[valid]

    rbins = np.arange(rmin, rmax, width)
    nbins = len(rbins[:-1])

    if method not in ("median", "mean"):
        raise ValueError("method must be 'median' or 'mean'")

    bin_low = rbins[:-1]
    bin_high = bin_low + width

    sr_col = sr[:, None]
    cond = (sr_col >= bin_low) & (sr_col <= bin_high)

    if method == "mean":
        counts = cond.sum(axis=0)
        sums = (sim[:, None] * cond).sum(axis=0)
        with np.errstate(invalid="ignore"):
            array = sums / counts
    else:  # median
        array = np.array([
            np.median(sim[cond[:, i]]) if np.any(cond[:, i]) else np.nan
            for i in range(nbins)
        ])

    return rbins[:-1] + 0.5 * width, array
