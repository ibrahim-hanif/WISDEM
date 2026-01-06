"""
utilities.py

Created by Andrew Ning on 2013-05-31. Copyright (c) NREL. All rights reserved.
Adapted by Vasudev Gupta on 2025-11-03. Dept. of Marine Technology, NTNU. All rights reserved.
"""
#(v) import libraries (global scope)
import logging
logger = logging.getLogger("wisdem/weis")

import numpy as np
from scipy.linalg import solve_banded

# from scipy.optimize import curve_fit
# ----- (v) below
import scipy.io as sio
from scipy.stats import weibull_min

def mode_fit(x, c2, c3, c4, c5, c6):
    return c2 * x**2.0 + c3 * x**3.0 + c4 * x**4.0 + c5 * x**5.0 + c6 * x**6.0


def get_modal_coefficients(x, y, deg=[2, 3, 4, 5, 6], idx0=None, base_slope0=True):
    if idx0 is None:
        idx0 = 0

    # Normalize x input
    xn = (x - x[idx0]) / (x[-1] - x[idx0])

    # Remove 0th and 1st order modes for base slope match
    if y.ndim > 1:
        y = y - y[idx0, np.newaxis, :]
        if not base_slope0:
            dy = np.gradient(y, xn, axis=0, edge_order=2)
            y = y - np.outer(xn, dy[idx0, :])
    else:
        y = y - y[idx0]
        if not base_slope0:
            dy = np.gradient(y, xn, edge_order=2)
            y = y - dy[idx0] * xn

    # Get coefficients to 2-6th order polynomial
    p6_0 = np.polynomial.polynomial.polyfit(xn, y, deg)

    # Normalize for Elastodyn
    # The normalization shouldn't be less than 1e-5 otherwise OpenFAST has trouble in single prec
    if y.ndim > 1:
        p6 = p6_0[2:, :]
        tempsum = np.sum(p6, axis=0) + 1e-16  # Avoid divide by 0
        normval = np.maximum(np.abs(tempsum), 1e-5)
        normval *= np.sign(tempsum)
        p6 /= normval[np.newaxis, :]
    else:
        p6 = p6_0[2:]
        tempsum = p6.sum() + 1e-16  # Avoid divide by 0
        normval = np.maximum(np.abs(tempsum), 1e-5)
        normval *= np.sign(tempsum)
        p6 /= normval

    return p6, p6_0


def get_xyz_mode_shapes(r, freqs, xdsp, ydsp, zdsp, xmpf, ympf, zmpf, idx0=None, base_slope0=True, expect_all=True):
    # Number of frequencies and modes
    nfreq = len(freqs)

    # Get mode shapes in batch
    mpfs = np.abs(np.c_[xmpf, ympf, zmpf])
    displacements = np.vstack((xdsp, ydsp, zdsp)).T
    polys, polys_raw = get_modal_coefficients(r, displacements, idx0=idx0, base_slope0=base_slope0)
    poly_d1 = np.polynomial.polynomial.polyder(polys_raw, m=1, axis=0)

    # Count the roots of the first derivative of the mode shape- First mode has zero, second has one, etc.
    xx = np.linspace(0.05, 1-1e-4, 1000) # BC at 0 can mess with root counting
    val_d1 = np.polynomial.polynomial.polyval(xx, poly_d1)
    nroot_d1 = np.count_nonzero(np.diff(np.sign(val_d1), axis=1), axis=1)
    
    xpolys = polys[:, :nfreq].T
    ypolys = polys[:, nfreq : (2 * nfreq)].T
    zpolys = polys[:, (2 * nfreq) :].T
    xroot1 = nroot_d1[:nfreq]
    yroot1 = nroot_d1[nfreq : (2 * nfreq)]
    #zroot1 = nroot_d1[(2 * nfreq) :]

    # Containers and counters for the mode shapes
    nfreq2 = int(nfreq / 2)
    mysize = nfreq2 if expect_all else nfreq
    mshapes_x = np.zeros((mysize, 5))
    mshapes_y = np.zeros((mysize, 5))
    mshapes_z = np.zeros((mysize, 5))
    freq_x = np.zeros(mysize)
    freq_y = np.zeros(mysize)
    freq_z = np.zeros(mysize)
    ix = 0
    iy = 0
    iz = 0

    # Identify which mode is which and whether it is a valid mode
    idir = np.argmax(mpfs, axis=1)
    mpfs_ratio = np.abs(mpfs.max(axis=1) / (1e-16 + mpfs.min(axis=1)))  # Avoid divide by 0

    for m in range(nfreq):
        if np.isnan(freqs[m]) or (freqs[m] < 1e-1) or (mpfs_ratio[m] < 1e3) or (mpfs[m, :].max() < 1e-13):
            continue
        
        if idir[m] == 0:
            if expect_all and ix >= nfreq2:
                continue
            imode = xroot1[m]
            if imode != ix and ix<2:
                logger.debug(f"WARNING: Freq no. {m}, x-dir: Mode numbder identified as {imode+1} going into slot {ix+1}")
            mshapes_x[ix, :] = xpolys[m, :]
            freq_x[ix] = freqs[m]
            ix += 1
        elif idir[m] == 1:
            if expect_all and iy >= nfreq2:
                continue
            imode = yroot1[m]
            if imode != iy and iy<2:
                logger.debug(f"WARNING: Freq no. {m}, y-dir: Mode numbder identified as {imode+1} going into slot {iy+1}")
            mshapes_y[iy, :] = ypolys[m, :]
            freq_y[iy] = freqs[m]
            iy += 1
        elif idir[m] == 2:
            if expect_all and iz >= nfreq2:
                continue
            # Torsional modes are not well captured by Frame3DD
            #imode = zroot1[m]
            #if imode != iz and iz<2:
            #    logger.debug(f"WARNING: Freq no. {m}, z-dir: Mode numbder identified as {imode+1} going into slot {iz+1}")
            mshapes_z[iz, :] = zpolys[m, :]
            freq_z[iz] = freqs[m]
            iz += 1
    '''
    # "Rank and file" the modeshapes by their mpfs and order
    # Filter the modeshapes by their mpfs
    #   - does guarauntees that modeshapes are calculated at different frequencies,
    #   - guarantees a modeshape in every direction
    #   - guarantees exact modeshape orders
    if rank_and_file:
        idyn = np.where(freqs > 1e-1)[0]
        freqs_dyn = freqs[idyn]
        ndyn = freqs_dyn.size
        eps = 1e-12
        dummy_span = np.linspace(0.0 + eps, 1.0 - eps, 100)
        defl_numbers = np.zeros((ndyn, 3))
        xmpf_dyn = np.abs(xmpf[idyn])
        ympf_dyn = np.abs(ympf[idyn])
        zmpf_dyn = np.abs(zmpf[idyn])
        xpolys_dyn = xpolys[idyn, :]
        ypolys_dyn = ypolys[idyn, :]
        zpolys_dyn = zpolys[idyn, :]
        polys_dyn = np.vstack((xpolys_dyn, ypolys_dyn, zpolys_dyn))
        pf = np.hstack((np.zeros((polys_dyn.shape[0], 2)),
                        polys_dyn))
        p_1deriv = np.polynomial.polynomial.polyder(pf, m=1, axis=1)
        p_2deriv = np.polynomial.polynomial.polyder(pf, m=2, axis=1)
        p_1val   = np.polynomial.polynomial.polyval(dummy_span, p_1deriv.T)
        p_2val   = np.polynomial.polynomial.polyval(dummy_span, p_2deriv.T)
        dnx1 = np.count_nonzero(np.diff(np.sign(p_1val), axis=1), axis=1)
        dnx2 = np.count_nonzero(np.diff(np.sign(p_2val), axis=1), axis=1)
        # Check second derivative for higher order modes
        for i in range(ndyn):
            for j in range(3):
                k = j*ndyn + i
                # Should only exist for higher order but monotonically increasing modes
                if dnx2[k] >= defl_numbers[i, j]:
                    defl_numbers[i, j] = dnx2[k] + 1
                else:
                    defl_numbers[i, j] = dnx1[k]
        
        #for j, polys in enumerate([xpolys, ypolys, zpolys]):
        #    poly_dyn = polys[idyn, :]
        #    for i, p in enumerate(poly_dyn):
        #        pf = np.r_[np.zeros(2), p]
        #        p_1deriv = np.polynomial.polynomial.polyder(pf, m=1, axis=0)
        #        p_1val = np.polynomial.polynomial.polyval(dummy_span, p_1deriv)
        #        dnx1 = np.count_nonzero(np.sign(p_1val[:-1]) != np.sign(p_1val[1:]))

        #        p_2deriv = np.polynomial.polynomial.polyder(pf, m=2, axis=0)
        #        p_2val = np.polynomial.polynomial.polyval(dummy_span, p_2deriv)
        #        dnx2 = np.count_nonzero(np.sign(p_2val[:-1]) != np.sign(p_2val[1:]))

        #        # Check second derivative for higher order modes
        #        if dnx2 >= defl_numbers[i, j]:  # Should only exist for higher order but monotonically increasing modes
        #            defl_numbers[i, j] = dnx2 + 1
        #        else:
        #            defl_numbers[i, j] = dnx1

        def record_used_freqs(polyidx, i, used_freq_idx):
            directions = ["x", "y", "z"]
            if polyidx in used_freq_idx and i < 3:
                logger.debug(
                    f"WARNING: Frequency index {polyidx} has been used again for i={i} in the {directions[i]}-direction"
                )
            used_freq_idx.append(polyidx)
            return used_freq_idx

        used_freq_idx = []
        for i in range(mysize):
            # Number of unique mode shape orders
            x_uniq_num = int(len(np.unique(defl_numbers[:, 0])) - 1)
            if i >= x_uniq_num:
                ix += 1
            # Get index of most dominant direction for i'th mode shape
            uniq_idx = min(i, x_uniq_num)  # use i'th mode shape, unless it doesn't exist, then use next largest
            mode_freq_idx = np.where(defl_numbers[:, 0] == np.unique(defl_numbers[:, 0])[uniq_idx])[
                0
            ]  # find frequency index where i'th mode shape exists
            x_polyidx = mode_freq_idx[
                np.argsort(-xmpf_dyn[mode_freq_idx])[min(ix, len(mode_freq_idx) - 1)]
            ]  # find index for i'th or the "next" i'th desired mode shape polynomial
            mshapes_x[i, :] = xpolys_dyn[x_polyidx, :]
            freq_x[i] = freqs_dyn[x_polyidx]
            used_freq_idx = record_used_freqs(x_polyidx, i, used_freq_idx)

            # repeat for y and z directions
            y_uniq_num = int(len(np.unique(defl_numbers[:, 1])) - 1)
            if i > y_uniq_num:
                iy += 1
            uniq_idx = min(i, y_uniq_num)  # use i'th mode shape, unless it doesn't exist, then use largest
            mode_freq_idx = np.where(defl_numbers[:, 1] == np.unique(defl_numbers[:, 1])[uniq_idx])[
                0
            ]  # find frequency index where i'th mode shape exists
            y_polyidx = mode_freq_idx[
                np.argsort(-ympf_dyn[mode_freq_idx])[min(iy, len(mode_freq_idx) - 1)]
            ]  # find index for i'th or the "next" i'th desired mode shape polynomial
            mshapes_y[i, :] = ypolys_dyn[y_polyidx, :]
            freq_y[i] = freqs_dyn[y_polyidx]
            used_freq_idx = record_used_freqs(y_polyidx, i, used_freq_idx)

            z_uniq_num = int(len(np.unique(defl_numbers[:, 2])) - 1)
            if i > z_uniq_num:
                iz += 1
            uniq_idx = min(i, z_uniq_num)  # use i'th mode shape, unless it doesn't exist, then use largest
            mode_freq_idx = np.where(defl_numbers[:, 2] == np.unique(defl_numbers[:, 2])[uniq_idx])[
                0
            ]  # find frequency index where i'th mode shape exists
            z_polyidx = mode_freq_idx[
                np.argsort(-zmpf_dyn[mode_freq_idx])[min(iz, len(mode_freq_idx) - 1)]
            ]  # find index for i'th or the "next" i'th desired mode shape polynomial
            mshapes_z[i, :] = zpolys_dyn[z_polyidx, :]
            freq_z[i] = freqs_dyn[z_polyidx]
            used_freq_idx = record_used_freqs(z_polyidx, i, used_freq_idx)
    '''

    return freq_x, freq_y, freq_z, mshapes_x, mshapes_y, mshapes_z


def rotate(xo, yo, xp, yp, angle):
    ## Rotate a point clockwise by a given angle around a given origin.
    # angle *= -1.
    qx = xo + np.cos(angle) * (xp - xo) - np.sin(angle) * (yp - yo)
    qy = yo + np.sin(angle) * (xp - xo) + np.cos(angle) * (yp - yo)
    return qx, qy


def arc_length(points):
    """
    Compute the distances between points along a curve and return those
    cumulative distances as a flat array.

    This function works for 2D, 3D, and N-D arrays.

    Parameters
    ----------
    points : numpy array[n_points, n_dimensions]
        Array of coordinate points that we compute the arc distances for.

    Returns
    -------
    arc_distances : numpy array[n_points]
        Array, starting at 0, with the cumulative distance from the first
        point in the points array along the arc.

    See Also
    --------
    wisdem.commonse.utilities.arc_length_deriv : computes derivatives for
    the arc_length function

    Examples
    --------
    Here is a simple example of how to use this function to find the cumulative
    distances between points on a 2D curve.

    >>> x_values = numpy.linspace(0., 5., 10)
    >>> y_values = numpy.linspace(2., 4., 10)
    >>> points = numpy.vstack((x_values, y_values)).T
    >>> arc_length(points)
    array([0.        , 0.59835165, 1.19670329, 1.79505494, 2.39340658,
           2.99175823, 3.59010987, 4.18846152, 4.78681316, 5.38516481])
    """
    cartesian_distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    arc_distances = np.r_[0.0, np.cumsum(cartesian_distances)]

    return arc_distances


def arc_length_deriv(points):
    """
    Return the Jacobian for function arc_length().
    See its docstring for more details.

    Parameters
    ----------
    points : numpy array[n_points, n_dim]
        Array of coordinate points that we compute the arc distances for.

    Returns
    -------
    d_arc_distances_d_points : numpy array[n_points, n_points * n_dim]
        Array, starting at 0, with the cumulative distance from the first
        point in the points array along the arc.
    """
    n_points, n_dim = points.shape
    d_arc_distances_d_points = np.zeros((n_points, n_points * n_dim))

    # Break out the two-line calculation into subparts to help obtain
    # derivatives easier
    diff_points = np.diff(points, axis=0)
    sum_diffs = np.sum(diff_points**2, axis=1)
    cartesian_distances = np.sqrt(sum_diffs)
    cum_sum_distances = np.cumsum(cartesian_distances)
    arc_distances = np.r_[0.0, cum_sum_distances]

    # This can be sped up slightly through numpy vectorization, it's shown here in
    # for-loop for clarity.
    for i in range(1, n_points):
        d_inner = 2 * points[i] - 2 * points[i - 1]
        d_outer = 0.5 * (sum_diffs[i - 1]) ** (-0.5)
        d_arc_distances_d_points[i, i * n_dim : i * n_dim + n_dim] = d_inner * d_outer

        d_inner = 2 * points[i - 1] - 2 * points[i]
        d_outer = 0.5 * (sum_diffs[i - 1]) ** (-0.5)
        d_arc_distances_d_points[i, i * n_dim - n_dim : i * n_dim] = d_inner * d_outer

        d_arc_distances_d_points[i:, i * n_dim - n_dim : i * n_dim] = np.sum(
            d_arc_distances_d_points[i - 1 : i + 1, i * n_dim - n_dim : i * n_dim], axis=0
        )

    return arc_distances, d_arc_distances_d_points


def cosd(value):
    """cosine of value where value is given in degrees"""

    return np.cos(np.radians(value))


def sind(value):
    """sine of value where value is given in degrees"""

    return np.sin(np.radians(value))


def tand(value):
    """tangent of value where value is given in degrees"""

    return np.tan(np.radians(value))


def hstack(vec):
    """stack arrays horizontally.  useful for assemblying Jacobian
    assumes arrays are column vectors (if rows just use concatenate)"""
    newvec = []
    for v in vec:
        if len(v.shape) == 1:
            newvec.append(v[:, np.newaxis])
        else:
            newvec.append(v)

    return np.hstack(newvec)


def vstack(vec):
    """stack arrays vertically
    assumes arrays are row vectors.  if columns use concatenate"""

    newvec = []
    for v in vec:
        if len(v.shape) == 1:
            newvec.append(v[np.newaxis, :])
        else:
            newvec.append(v)

    return np.vstack(newvec)


def _checkIfFloat(x):
    try:
        n = len(x)
    except TypeError:  # if x is just a float
        x = np.array([x])
        n = 1

    return x, n


def linspace_with_deriv(start, stop, num):
    """creates linearly spaced arrays, and derivatives for changing end points"""

    step = (stop - start) / float((num - 1))
    y = np.arange(0, num) * step + start
    y[-1] = stop

    # gradients
    const = np.arange(0, num) * 1.0 / float((num - 1))
    dy_dstart = -const + 1.0
    dy_dstart[-1] = 0.0

    dy_dstop = const
    dy_dstop[-1] = 1.0

    return y, dy_dstart, dy_dstop


def sectional_interp(xi, x, y):
    epsilon = 1e-11
    xx = np.c_[x[:-1], x[1:] - epsilon].flatten()
    yy = np.c_[y, y].flatten()
    return np.interp(xi, xx, yy)


def sectionalInterp(xi, x, y):
    return sectional_interp(xi, x, y)


def interp_with_deriv(x, xp, yp):
    """linear interpolation and its derivative. To be precise, linear interpolation is not
    differentiable right at the control points, but in general it works well enough"""
    # TODO: put in Fortran to speed up

    x, n = _checkIfFloat(x)

    if np.any(np.diff(xp) < 0):
        raise TypeError("xp must be in ascending order")

    # n = len(x)
    m = len(xp)

    y = np.zeros(n)
    dydx = np.zeros(n)
    dydxp = np.zeros((n, m))
    dydyp = np.zeros((n, m))

    for i in range(n):
        if x[i] < xp[0]:
            j = 0  # linearly extrapolate
        elif x[i] > xp[-1]:
            j = m - 2
        else:
            for j in range(m - 1):
                if xp[j + 1] > x[i]:
                    break
        x1 = xp[j]
        y1 = yp[j]
        x2 = xp[j + 1]
        y2 = yp[j + 1]

        y[i] = y1 + (y2 - y1) * (x[i] - x1) / (x2 - x1)
        dydx[i] = (y2 - y1) / (x2 - x1)
        dydxp[i, j] = (y2 - y1) * (x[i] - x2) / (x2 - x1) ** 2
        dydxp[i, j + 1] = -(y2 - y1) * (x[i] - x1) / (x2 - x1) ** 2
        dydyp[i, j] = 1 - (x[i] - x1) / (x2 - x1)
        dydyp[i, j + 1] = (x[i] - x1) / (x2 - x1)

    if n == 1:
        y = y[0]

    return y, np.diag(dydx), dydxp, dydyp


def assembleI(I):
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I[0], I[1], I[2], I[3], I[4], I[5]
    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


def unassembleI(I):
    return np.r_[I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]


def rotateI(I, th, axis="z"):
    # https://en.wikipedia.org/wiki/Moment_of_inertia
    # With rotation matrix R, then I = R * I_0 * R^T

    # Build inertia tensor
    if I.ndim == 2 and I.shape[0] == 3 and I.shape[1] == 3:
        Iin = unassemble(I)
    elif I.ndim == 1 and I.size == 3:
        Iin = np.r_[I, np.zeros(3)]
    elif I.ndim == 1 and I.size == 6:
        Iin = I.copy()
    else:
        raise ValueError("Unknown size for input, I:", I)
    Imat = assembleI(Iin)

    # Build rotation matrix
    ct = np.cos(th)
    st = np.sin(th)
    if axis in ["z", "Z", 2]:
        R = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    elif axis in ["y", "Y", 1]:
        R = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    elif axis in ["x", "X", 0]:
        R = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    else:
        raise ValueError("Axis must be either x/y/z or 0/1/2")

    Iout = unassembleI(R @ Imat @ R.T)

    return Iout


def rotate_align_vectors(a, b):
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    unita = a / mag_a
    unitb = b / mag_b
    v = np.cross(unita, unitb)
    s = np.linalg.norm(v)
    c = np.dot(unita, unitb)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    radd = np.zeros((3, 3)) if s < 1e-6 else vx + np.dot(vx, vx) / (1 + c)  # *(1-c)/(s**2) #
    r = np.eye(3) + radd
    return r


def cubic_with_deriv(x, xp, yp):
    """deprecated"""

    x, n = _checkIfFloat(x)

    if np.any(np.diff(xp) < 0):
        raise TypeError("xp must be in ascending order")

    # n = len(x)
    m = len(xp)

    y = np.zeros(n)
    dydx = np.zeros(n)
    dydxp = np.zeros((n, m))
    dydyp = np.zeros((n, m))

    xk = xp[1:-1]
    yk = yp[1:-1]
    xkp = xp[2:]
    ykp = yp[2:]
    xkm = xp[:-2]
    ykm = yp[:-2]

    b = (ykp - yk) / (xkp - xk) - (yk - ykm) / (xk - xkm)
    l = (xk - xkm) / 6.0
    d = (xkp - xkm) / 3.0
    u = (xkp - xk) / 6.0
    # u[0] = 0.0  # non-existent entries
    # l[-1] = 0.0

    # solve for second derivatives
    fpp = solve_banded((1, 1), np.array([u, d, l]), b)
    fpp = np.concatenate([[0.0], fpp, [0.0]])  # natural spline

    # find location in vector
    for i in range(n):
        if x[i] < xp[0]:
            j = 0
        elif x[i] > xp[-1]:
            j = m - 2
        else:
            for j in range(m - 1):
                if xp[j + 1] > x[i]:
                    break
        x1 = xp[j]
        y1 = yp[j]
        x2 = xp[j + 1]
        y2 = yp[j + 1]

        A = (x2 - x[i]) / (x2 - x1)
        B = 1 - A
        C = 1.0 / 6 * (A**3 - A) * (x2 - x1) ** 2
        D = 1.0 / 6 * (B**3 - B) * (x2 - x1) ** 2

        y[i] = A * y1 + B * y2 + C * fpp[j] + D * fpp[j + 1]
        dAdx = -1.0 / (x2 - x1)
        dBdx = -dAdx
        dCdx = 1.0 / 6 * (3 * A**2 - 1) * dAdx * (x2 - x1) ** 2
        dDdx = 1.0 / 6 * (3 * B**2 - 1) * dBdx * (x2 - x1) ** 2
        dydx[i] = dAdx * y1 + dBdx * y2 + dCdx * fpp[j] + dDdx * fpp[j + 1]

    if n == 1:
        y = y[0]
        dydx = dydx[0]

    return y


def trapz_deriv(y, x):
    """trapezoidal integration and derivatives with respect to integrand or variable."""

    dI_dy = np.gradient(x)
    dI_dy[0] /= 2
    dI_dy[-1] /= 2

    dI_dx = -np.gradient(y)
    dI_dx[0] = -0.5 * (y[0] + y[1])
    dI_dx[-1] = 0.5 * (y[-1] + y[-2])

    return dI_dy, dI_dx


def _smooth_maxmin(yd, ymax, maxmin, pct_offset=0.01, dyd=None):
    yd, n = _checkIfFloat(yd)

    y1 = (1 - pct_offset) * ymax
    y2 = (1 + pct_offset) * ymax

    dy1 = 1 - pct_offset
    dy2 = 1 + pct_offset

    if maxmin == "min":
        f1 = y1
        f2 = ymax
        g1 = 1.0
        g2 = 0.0
        idx_constant = yd >= y2

        df1 = dy1
        df2 = 1.0

    elif maxmin == "max":
        f1 = ymax
        f2 = y2
        g1 = 0.0
        g2 = 1.0
        idx_constant = yd <= y1

        df1 = 1.0
        df2 = dy2

    f = CubicSplineSegment(y1, y2, f1, f2, g1, g2)

    # main region
    ya = np.copy(yd)
    if dyd is None:
        dya_dyd = np.ones_like(yd)
    else:
        dya_dyd = np.copy(dyd)

    dya_dymax = np.zeros_like(ya)

    # cubic spline region
    idx = np.logical_and(yd > y1, yd < y2)
    ya[idx] = f.eval(yd[idx])
    dya_dyd[idx] = f.eval_deriv(yd[idx])
    dya_dymax[idx] = f.eval_deriv_params(yd[idx], dy1, dy2, df1, df2, 0.0, 0.0)

    # constant region
    ya[idx_constant] = ymax
    dya_dyd[idx_constant] = 0.0
    dya_dymax[idx_constant] = 1.0

    if n == 1:
        ya = ya[0]
        dya_dyd = dya_dyd[0]
        dya_dymax = dya_dymax[0]

    return ya, dya_dyd, dya_dymax


def smooth_max(yd, ymax, pct_offset=0.01, dyd=None):
    """array max, uses cubic spline to smoothly transition.  derivatives with respect to array and max value.
    width of transition can be controlled, and chain rules for differentiation"""
    return _smooth_maxmin(yd, ymax, "max", pct_offset, dyd)


def smooth_min(yd, ymin, pct_offset=0.01, dyd=None):
    """array min, uses cubic spline to smoothly transition.  derivatives with respect to array and min value.
    width of transition can be controlled, and chain rules for differentiation"""
    return _smooth_maxmin(yd, ymin, "min", pct_offset, dyd)


def smooth_abs(x, dx=0.01):
    """smoothed version of absolute value function, with quadratic instead of sharp bottom.
    Derivative w.r.t. variable of interest.  Width of quadratic can be controlled"""

    x, n = _checkIfFloat(x)

    y = np.abs(x)
    idx = np.logical_and(x > -dx, x < dx)
    y[idx] = x[idx] ** 2 / (2.0 * dx) + dx / 2.0

    # gradient
    dydx = np.ones_like(x)
    dydx[x <= -dx] = -1.0
    dydx[idx] = x[idx] / dx

    if n == 1:
        y = y[0]
        dydx = dydx[0]

    return y, dydx


def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


def closest_node(nodemat, inode):
    if not nodemat.shape[1] in [2, 3]:
        if nodemat.shape[0] in [2, 3]:
            xyz = nodemat.T
        else:
            raise ValueError("Expected an m X 2/3 input node array")
    else:
        xyz = nodemat

    if not len(inode) in [2, 3]:
        raise ValueError("Expected a size 2 or 3 node point")

    return np.sqrt(np.sum((xyz - inode[np.newaxis, :]) ** 2, axis=1)).argmin()


def nodal2sectional(x, axis=0):
    """Averages nodal data to be length-1 vector of sectional data

    INPUTS:
    ----------
    x   : float vector, nodal data

    OUTPUTS:
    -------
    y   : float vector,  sectional data
    """
    if x.ndim == 1:
        y = 0.5 * (x[:-1] + x[1:])
        dy = np.c_[0.5 * np.eye(y.size), np.zeros(y.size)]
        dy[np.arange(y.size), 1 + np.arange(y.size)] = 0.5
    elif x.ndim == 2 and axis == 0:
        y = 0.5 * (x[:-1, :] + x[1:, :])
        dy = None
    elif x.ndim == 2 and axis == 1:
        y = 0.5 * (x[:, :-1] + x[:, 1:])
        dy = None
    else:
        raise ValueError("Only 2 dimensions supported")

    return y, dy


def sectional2nodal(x):
    return np.r_[x[0], np.convolve(x, [0.5, 0.5], "valid"), x[-1]]


def cubic_spline_eval(x1, x2, f1, f2, g1, g2, x):
    spline = CubicSplineSegment(x1, x2, f1, f2, g1, g2)
    return spline.eval(x)


class CubicSplineSegment(object):
    """cubic splines and the their derivatives with with respect to the variables and the parameters"""

    def __init__(self, x1, x2, f1, f2, g1, g2):
        self.x1 = x1
        self.x2 = x2

        self.A = np.array(
            [
                [x1**3, x1**2, x1, 1.0],
                [x2**3, x2**2, x2, 1.0],
                [3 * x1**2, 2 * x1, 1.0, 0.0],
                [3 * x2**2, 2 * x2, 1.0, 0.0],
            ]
        )
        self.b = np.array([f1, f2, g1, g2])

        self.coeff = np.linalg.solve(self.A, self.b)

        self.poly = np.polynomial.Polynomial(self.coeff[::-1])

    def eval(self, x):
        return self.poly(x)

    def eval_deriv(self, x):
        polyd = self.poly.deriv()
        return polyd(x)

    def eval_deriv_params(self, xvec, dx1, dx2, df1, df2, dg1, dg2):
        x1 = self.x1
        x2 = self.x2
        dA_dx1 = np.array(
            [[3 * x1**2, 2 * x1, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [6 * x1, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        dA_dx2 = np.array(
            [[0.0, 0.0, 0.0, 0.0], [3 * x2**2, 2 * x2, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [6 * x2, 2.0, 0.0, 0.0]]
        )
        df = np.array([df1, df2, dg1, dg2])
        c = np.array(self.coeff).T

        n = len(xvec)
        dF = np.zeros(n)
        for i in range(n):
            x = np.array([xvec[i] ** 3, xvec[i] ** 2, xvec[i], 1.0])
            d = np.linalg.solve(self.A.T, x)
            dF_dx1 = -d @ dA_dx1 @ c
            dF_dx2 = -d @ dA_dx2 @ c
            dF_df = np.linalg.solve(self.A.T, x)
            dF[i] = np.dot(dF_df, df) + dF_dx1 * dx1 + dF_dx2 * dx2

        return dF

#%%[markdown]
# Vasudev Gupta: adaptations
# ============================================================
#%%
def get_recorder_results(filename, list_driver_vars=None, dvs_as_arrays=False):
    """
    Inputs:
    -------
    filename : str
        path to the sqlite recorder file
    list_driver_vars : dict
        output of prob.driver.list_vars() openmdao function
    dvs_as_arrays : bool
        if (some) design variables are arrays (size>1), set to True
    
    Outputs:
    -------
    results
        if dvs_as_arrays is False (default):
            np.array, size (n_cases, n_DVs + n_objs)
            array with rows as cases and columns as DVs followed by objectives
        else:
            dict of np.arrays
            with keys as [DVs, constrs, objs], each array size (n_cases, array_size)

    Internal Progress:
    -----------------
    - DONE : change from predefined names (x,y,obj) to input list_driver_vars (or DrivetrainSE specific)
    - DONE : adopt for array (size>1) DVs like lss_*
    """

    import openmdao.api as om

    if not dvs_as_arrays: # (default) DVs as scalars, return np.array

        if list_driver_vars is not None:
            isolate = ['objectives', 'design_vars']
            name_obj = [list_driver_vars[isolate[0]][0][0]] 
            name_DVs = []
            for i in range( len( list_driver_vars[isolate[1]] ) ):
                name = list_driver_vars[isolate[1]][i][0]
                name_DVs.append( name )
                # val = prob.get_val( name )
                # print(f'DV: {name} = {val}')
        else:
            name_obj = ['nacelle_mass']
            name_DVs = ['L_h1', 'L_12', 'lss_diameter', 'lss_wall_thickness']
            # TODO: adopt for array (size>1) DVs like lss_*

        # case reader accessing
        cr = om.CaseReader(filename)
        cases = cr.list_cases('driver', out_stream=None) #NOTE:cases are a list of Case objects; with 'outputs' attribute
        len_cases = len(cases)

        results = np.zeros( (len_cases, (len(name_DVs)+len(name_obj)) ) ) # ( dv(1), dv(2), ..., obj )
        for i in range(len_cases):
            case = cases[i]
            output = cr.get_case(case).outputs
            dvs = [float(output[ name_DVs[i] ][0]) for i in range(len(name_DVs))]
            results[i,:] = np.r_[ dvs, float(output[name_obj[0]][0]) ]
        
        # results = np.squeeze(np.array(results)).reshape( (length, 3) ) # size (n, 3)
    
    else: # DVs as arrays, return dict (of np.arrays)
        results = {}
        
        # case reader accessing
        cr = om.CaseReader(filename)
        cases = cr.list_cases('driver', out_stream=None) #NOTE:cases are a list of Case objects; with 'outputs' attribute
        len_cases = len(cases)
        
        for i in range(len_cases):
            out = cr.get_case(i).outputs
            # rd = {**rd, **out}
            for key, val in out.items():
                if key in results:
                    results[key] = np.vstack( (results[key], val) )
                else:
                    results[key] = val

    return results
    """
    # results dictionary
    d = {}
    # init all DVs into results dictionary
    dvs = prob.model.get_design_vars()
    names_DVs = []
    for key, val in dvs.items():
        print( key )
        names_DVs.append( key )
        print( val['size'] )
        d[key] = np.zeros( (len_cases,val['size']) )
    # init all Objs into results dictionary
    objs = prob.model.get_objectives()
    names_obj = []
    for key, val in objs.items():
        print( key )
        names_obj.append( key )
        print( val['size'] )
        d[key] = np.zeros( (len_cases,val['size']) )
    d
    """

# ---------------
def mainshaft_loads_from_mat_to_dict( loc_FLS_loads_mat_file, loc_ULS_loads_mat_file, loc_all_loads_mat_file="" ):
    """
    Inputs:
    -------
    loc_FLS_loads_mat_file : str
        path to FLS loads .mat file
    loc_FLS_loads_mat_file : str
        path to ULS loads .mat file
    loc_all_loads_mat_file : str
        path to all loads .mat file (named `mainshaft_loads.mat`)
    
    Outputs:
    -------
    Snew : dict
        with keys = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        and values= np.array of shape (720000,10)
        &
        with keys = ['Fx_max','Fy_max','Fz_mean', 'Mx_max','My_max','Mz_max']
        and values= float

    Internal Progress:
    -----------------
    - DONE : implementation
    """
    # init
    print(f' --- Loading: mainshaft FLS loads from {loc_FLS_loads_mat_file}')
    print(f' --- Loading: mainshaft ULS loads from {loc_ULS_loads_mat_file}')
    if len(loc_all_loads_mat_file) != 0: print(f' --- Loading: all mainshaft loads from {loc_all_loads_mat_file}')

    try:
        S_FLS = sio.loadmat(loc_FLS_loads_mat_file)
        S_ULS = sio.loadmat(loc_ULS_loads_mat_file)
        if len(loc_all_loads_mat_file) != 0: S_all = sio.loadmat(loc_all_loads_mat_file)

    except Exception as e:
        raise RuntimeError(f'Could not load FLS/ULS .mat files: {e}')
    # Merge both into unified structure (SI units: N, Nm)
    S = {
        'Fx': S_FLS['S_FLS']['Fx'][0, 0], # S['Fx'].shape=(1,10); S['Fx'][0,i] being (72e4,1) per ws
        'Fy': S_FLS['S_FLS']['Fy'][0, 0],
        'Fz': S_FLS['S_FLS']['Fz'][0, 0],
        'Mx': S_FLS['S_FLS']['Mx'][0, 0],
        'My': S_FLS['S_FLS']['My'][0, 0],
        'Mz': S_FLS['S_FLS']['Mz'][0, 0],
        'ws': S_FLS['S_FLS']['ws'][0,0],       # np.array of shape (1,10)
        'Time': S_FLS['S_FLS']['Time'][0, 0], # np.array of shape (720000,10)
        'rot_speed': S_FLS['S_FLS']['rot_speed'][0, 0], # np.array of shape (720000,10)

        'Fx_max': S_ULS['S_ULS']['Fx_max'][0, 0].item(),
        'Fy_max': S_ULS['S_ULS']['Fy_max'][0, 0].item(),
        'Fz_mean': S_ULS['S_ULS']['Fz_mean'][0, 0].item(), #TODO: "_mean" alongisde "_max"! why?! + automate with openFAST outputs
        'Mx_max': S_ULS['S_ULS']['Mx_max'][0, 0].item(),
        'My_max': S_ULS['S_ULS']['My_max'][0, 0].item(),
        'Mz_max': S_ULS['S_ULS']['Mz_max'][0, 0].item(),
    }
    print(' --- Success: loaded FLS and ULS load data.\n')

    # init
    n_timeSteps = S['Fx'][0,0].shape[0]
    n_ws = S['Fx'].shape[1]

    Snew = {}
    keys_all = []
    for key,val in S.items():
        # init
        # (0) key == 'Fx', val.shape == (1,10)
        keys_all.append(key)

        # 1. float (ULS load)
        if key.endswith('_max') or key.endswith('_mean'):
            Snew[ key ] = float( val )
        # 2. vector (ws)
        elif key == 'ws':
            Snew[ key ] = val.reshape( 1,n_ws )
        elif key == 'Time': # time values same for all ws, save as vector
            Snew[ key ] = val[0,0].reshape( n_timeSteps,1 )
        # 3. matrix (FLS load & rot_speed)
        else:
            Snew[ key ] = np.zeros(( n_timeSteps,n_ws  ))    
            # print( Snew[key].shape ) # for debugging
            for i in range( n_ws ):
                Snew[ key ][:,i] = val[0,i].reshape( n_timeSteps, )

    print(' --- Success: converted .mat loads to dict (returning).\n')
    return Snew, keys_all

# ---------------
def load_all_mat_to_dict( loc_all_loads_mat_file ):
    """
    Inputs:
    -------
    loc_all_loads_mat_file : str
        path to all loads .mat file (named `mainshaft_loads.mat`)

    Outputs:
    -------
    Snew : dict
        keys    = ['F*','F*_max', 'M*','M*_max', 'Time', 'mean_wind_speed','rot_speed']
        values  = np.array[720000,10] for FLS loads & rot_speed
                = np.array[1,10] for ULS loads & others
    keys_all : list
        list of all keys in Snew
    """
    # load database
    print(f' --- Loading: all mainshaft loads from {loc_all_loads_mat_file}')
    try:
        S_all = sio.loadmat(loc_all_loads_mat_file)
    except Exception as e:
        raise RuntimeError(f'Could not load from the .mat file: {e}')

    # compute
    n_timeSteps,n_ws = S_all['Fx'].shape[1], S_all['Fx'].shape[0]

    Snew = {}
    keys_all = []
    for key,val in S_all.items():
        # init
        # (0) key == 'Fx', val.shape == (1,10)
        
        if key.startswith('__'): continue

        if key.endswith('_max') or (                            # ULS load
                key == 'Time') or (key == 'mean_wind_speed'):   # others
            keys_all.append(key)
            Snew[ key ] = val
        elif key.endswith('_mean') or key.endswith('_st'): continue

        if val.shape[0] > 1:                    # FLS load & rot_speed
            keys_all.append(key)
            if val.shape[1] > val.shape[0]:
                Snew[ key ] = val.T    
            # Snew[ key ] = np.zeros(( n_timeSteps,n_ws  ))    
            # print( Snew[key].shape ) # for debugging
            else: Snew[ key ].shape = val

    print(' --- Success: converted .mat all-loads to dict (returning).\n')
    return Snew, keys_all

# ---------------
def pdf_norm_int_using_cdf( ws_pts, coeff_weibull=(1.95,11.6) ):
    """
    Compute normalized PDF of Weibull distribution at given (ws) points.

    Inputs
    -------
    ws_pts : array[1,n]
        Points at which to evaluate the PDF.
        Typically wind speed bins, [5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25] [m/s]
    coeff_weibull : tuple
        Coefficients of Weibull distribution for site (c, scale)
        Typically for utsira, (1.95, 11.6)
    
    Outputs
    -------
    pdf_vals : array[n]
        Normalized PDF values at the given points.
    """
    # init
    # force ws to be 1D array of shape (1,n)
    ws_pts = np.atleast_2d(ws_pts)
    
    # compute
    delta = ws_pts[0,1] - ws_pts[0,0] #NOTE: 2D array of shape (1,n) with just one row index=0
    ws_ranges = np.array((
        np.r_[ ws_pts[0,0], ws_pts[0,1:]-(delta / 2)],
        np.r_[ (delta/2) + ws_pts[0,:-1], ws_pts[0,-1] ],
    ))

    pdf = (
        weibull_min.cdf(ws_ranges[1,:], 1.95, scale=11.6) -
        weibull_min.cdf(ws_ranges[0,:], 1.95, scale=11.6)
    )
    # pdf_vals = pdf / np.sum(pdf) #TODO: check if this is needed
    return pdf

# ---------------
def bin_counting_of_load(load_series, ws, probabilities, 
                         p=10/3, nBins=100):
    """
    Compute equivalent load using bin counting method (histogram).
    NOTE: BINNING BAD WITHIN OPTIMIZATION, TODO: DOCUMENT FAILURE AND SOLUTION!

    Inputs
    -------
    load_series : array[ # of time steps , # of wind speeds ]
        Time series of loads (forces or moments); for FLS, shape=(72e4,10)
    ws : array[ 1, # of wind speeds ]
        Wind speed bins corresponding to load_series columns, shape=(1,10)
    probabilities : array[ 1, # of wind speeds ]
        Probability of occurance of each wind speed
    p : float
        Exponent for equivalent load calculation (default: 10/3 for roller bearings)
    nBins : int
        Number of bins to use for histogramming the load data
    
    Outputs
    -------
    load_eq : array[ # of wind speeds ]
        Equivalent load per wind speed bin, shape=(10,)

    Internal Progress
    ______
    - DONE : implementation
    - DONE : replace coeff_weibull with ws probabilities directly as input
    - TODO : CORRECT this wrong implementation 
    """
    # init
    P = load_series  # shape=(72e4,10)
    n_t, n_w = P.shape[0], P.shape[1] # 72e3, 11
    # ws_pdf = pdf_norm_int_using_cdf(ws, coeff_weibull) # (1,10)
    ws_pdf = probabilities.reshape(1,n_w)
    
    Pmax = np.max(P) if np.max(P) > 0 else np.finfo(float).eps
    
    edges_P = np.linspace(0, Pmax, nBins + 1)
    centers_P = (edges_P[:-1] + edges_P[1:]) / 2
    
    P_hist = np.zeros(nBins)    
    for ec in range(len(ws)):
        hist_count, _ = np.histogram(P[:, ec], bins=edges_P, density=True) #density, for it is PDF
        # hist_count = hist_count / np.sum(hist_count) if np.sum(hist_count) > 0 else hist_count #divide to normalize (sum=1), not needed coz density=True
        P_hist += ws_pdf[0,ec] * hist_count

    P_sum = (np.sum((centers_P ** (p)) * P_hist)) ** (1/p)
    return P_sum

# ---------------
def scale_bounds_for_driver( lb, ub ):
    """
    function to provide the m (scalar) and b (adder) values for uniform
    scaling of openMDAO driver, with which
    - lb --> 0.0
    - ub --> 1.0
    with formula: x_driver = m * x_model + b
    """
    ref0, ref = lb, ub
    m = 1 / (ref-ref0)
    b = -ref0
    return m,b