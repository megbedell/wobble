# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["fit_continuum"]

import numpy as np


def fit_continuum(x, y, order=3, nsigma=3.0, maxniter=50):
    """Fit the continuum using sigma clipping

    Args:
        x: The wavelengths
        y: The log-fluxes
        order: The polynomial order to use
        nsigma: The sigma clipping threshold
        maxniter: The maximum number of iterations to do

    Returns:
        The value of the continuum at the wavelengths in x

    """
    A = np.vander(x - np.mean(x), order+1)
    m = np.ones(len(x), dtype=bool)
    for i in range(maxniter):
        w = np.linalg.solve(np.dot(A[m].T, A[m]), np.dot(A[m].T, y[m]))
        mu = np.dot(A, w)
        resid = y - mu
        sigma = np.sqrt(np.median(resid**2))
        m_new = np.abs(resid) < nsigma*sigma
        if m.sum() == m_new.sum():
            m = m_new
            break
        m = m_new
    return mu
