#!/usr/bin/env python
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as np_pi, sin as np_sin, cos as np_cos, exp as np_exp

"""
See equations in section 5 in Strateva et al. (2003).
Note that there is a typo in their Lorentz factor.
Use the correct equation in Eracleous et al. (1995) instead.
The line profiles are expressed in the velocity domain.
The derivation can be found in the ipynb file.
Implemented by TH on 08/16/2019.
"""

c = 3e5  # km/s
normdist_renorm = 1.0 / (2.0 * np_pi)**0.5


def int_func_op(phi, xi, X, q, V_sig, inc, ecc, phi0, intnorm):
    """The integrand F_X.

    Parameters
    ----------
    phi : float
        azimuthal angle of the disk in radians.
    xi : float
        a dimensionless distance from the black hole in units of Rg.
    X : float
        velocity in units of c (speed of light).
    q : float
        the power-law index of surface emissivity.
    V_sig : float
        velocity broadening parameter in km/s.
    inc : float
        inclination angle in radians.
    ecc : float
        eccentricity of the disk. 0 <= ecc < 1 to ensure a closed orbit.
    phi0 : float
        orientation angle (radians) of the disk relative to
        the observers line-of-sight.
    intnorm : float
        a normalization factor to alleviate numerical issues.

    Returns
    -------
    float
        specific intensity of the disk emission evaluated at velocity X.
    """
    sininc = np_sin(inc)
    sininccosphi = sininc * np_cos(phi)
    sinincsinphi = sininc * np_sin(phi)
    eccsinphiphi0 = ecc * np_sin(phi - phi0)
    BBsq = 1. - ecc * np_cos(phi - phi0)
    xform = (1. + ecc) / BBsq
    xifunc = xi * xform
    psival = 1. + (1. - sininccosphi) / (1. + sininccosphi) / xifunc
    AA = 1. - 2. / xifunc
    CCsq = 1. - sininccosphi**2
    # Has the leading order term of (b/r) approximation...
    termBinner = 1. - psival**2 * CCsq * AA
    if termBinner < 0:
        # ...zero it if it becomes negative
        # (affects some values of phi when xi is small)
        termBinner = 0
    termB = (termBinner / (xifunc * AA**3 * BBsq))**0.5 * eccsinphiphi0
    termC = psival * sinincsinphi * (BBsq / (xifunc - 2.))**0.5
    lorentzf = (1. - ((eccsinphiphi0 / AA)**2 / xifunc /
                      BBsq + (BBsq / (xifunc - 2.))))**-0.5
    D_invval = lorentzf * (AA**-0.5 - termB + termC)
    exponent = -0.5 * ((((1. + X) * D_invval - 1.) * (c / V_sig))**2)

    if exponent > -37:
        # Truncate this at machine precision.
        # The contributions are dominated by where this exponent is ~0,
        # keeping arbitrarily small numbers here just leads to more
        # round off/problems in the integrator backend.
        return intnorm * (
            xifunc**(1. - q) * xform * np_exp(exponent)
        ) * psival / D_invval**3
    else:
        return 0.0


def e_model(theta, x):
    """Computes the elliptical disk line profile in velocity space.

    Parameters
    ----------
    theta : list
        a list of floats that contains 7 parameters describing
        the thin elliptical disk.
    x : list
        a list of velocities in floats where the specific intensity I(v)
        is evaluated.

    Returns
    -------
    ndarray
        1d array that contains normalized specific intensity I(v)
        at each velocity in array x.
    """
    q, sig, incl, ecc, phi0, r1, r2 = theta
    intnorm = normdist_renorm / sig
    # The negative sign (-X) is due to
    # the change from frequency to velocity space (dnu = -dv).
    md = np.array([integrate.dblquad(int_func_op, r1, r2, 0., 1.999999 * np_pi,
                                     args=(-X, q, sig, incl,
                                           ecc, phi0, intnorm))[0] for X in x])
    # Normalize model profile to data
    md /= np.max(md)
    return md


def model_validate_phi0():
    """Reproduce Fig 3a in Eracleous (1995)...
    """
    print("Reproduce Fig 3a in Eracleous (1995)")
    print("Disk line profiles with varying orientation angle")
    X_grid = np.arange(-0.04, 0.05, 0.0011)
    sig, incl, ecc, phi0, r1, r2 = [
        800, np_pi / 6, 0.3, 0.5 * np_pi, 500, 2500]
    offset = 0
    fig, ax = plt.subplots(figsize=(4.5, 6))
    for ind, phi0_deg in enumerate(np.arange(0, 324.01, 36)):
        phi0 = (phi0_deg / 180) * np_pi
        y_new = e_model([3, sig, incl, ecc,
                         phi0, r1, r2], X_grid)
        plt.plot((X_grid + 1) * 6564, y_new + offset,
                 color='C%d' % (ind), ls='solid', label="", lw=0.7)
        plt.text(6800, offset, r"$\phi_0$=%d$^\circ$" % (phi0_deg))
        offset += 0.5
    plt.axvline(6564, linestyle='dashed', color='grey')
    plt.show()
    plt.close()


def model_validate_ecc():
    """Reproduce Fig 3b in Eracleous (1995)...
    """
    print("Reproduce Fig 3b in Eracleous (1995)")
    print("Disk line profiles with varying eccentricity")
    X_grid = np.arange(-0.04, 0.05, 0.0011)
    sig, incl, ecc, phi0, r1, r2 = [
        800, np_pi / 6, 0.3, 0.5 * np_pi, 500, 2500]
    offset = 0
    fig, ax = plt.subplots(figsize=(4.5, 6))
    for ind, ecc in enumerate(np.arange(0.1, 0.551, 0.05)):
        # y_new = e_model([3, sig, incl, ecc, phi0, r1, r2], X_grid)
        y_new = e_model([3, sig, incl, ecc, phi0, r1, r2], X_grid)
        # yy.append(y_new)
        plt.plot((X_grid + 1) * 6564, y_new + offset,
                 color='C%d' % (ind), ls='solid', label="", lw=0.7)
        plt.text(6800, offset, r"e=%.2f" % (ecc))
        offset += 0.3
    plt.axvline(6564, linestyle='dashed', color='grey')
    plt.show()
    plt.close()


def model_validate_rin():
    """Reproduce Fig 3c in Eracleous (1995)...
    """
    print("Reproduce Fig 3c in Eracleous (1995)")
    print("Disk line profiles with varying inner disk radius")
    X_grid = np.arange(-0.04, 0.05, 0.0011)
    sig, incl, ecc, phi0, r1, r2 = [
        800, np_pi / 6, 0.3, 0.5 * np_pi, 500, 2500]
    offset = 0
    fig, ax = plt.subplots(figsize=(4.5, 6))
    for ind, r1 in enumerate(np.arange(200, 1000.01, 100)[::-1]):
        y_new = e_model([3, sig, incl, ecc,
                         phi0, r1, r2], X_grid)
        plt.plot((X_grid + 1) * 6564, y_new + offset,
                 color='C%d' % (ind), ls='solid', label="", lw=0.7)
        plt.text(6800, offset, r"$\xi_{1}$=%d" % (r1))
        offset += 0.5
    plt.axvline(6564, linestyle='dashed', color='grey')
    plt.show()
    plt.close()
