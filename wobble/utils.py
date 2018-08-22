# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import tensorflow as tf
import sys

speed_of_light = 2.99792458e8   # m/s

__all__ = ["fit_continuum", "bin_data", "get_session", "doppler"]

def get_session(restart=False):
  """Get the globally defined TensorFlow session.
  If the session is not already defined, then the function will create
  a global session.
  Returns:
    _SESSION: tf.Session.
  (Code from edward package.)
  """
  global _SESSION
  if tf.get_default_session() is None:
    _SESSION = tf.InteractiveSession()
    
  else:
    _SESSION = tf.get_default_session()

  if restart:
      _SESSION.close()
      _SESSION = tf.InteractiveSession()

  save_stderr = sys.stderr
  return _SESSION
  
def doppler(v, tensors=True):
    frac = (1. - v/speed_of_light) / (1. + v/speed_of_light)
    if tensors:
        return tf.sqrt(frac)
    else:
        return np.sqrt(frac)


def fit_continuum(x, y, ivars, order=6, nsigma=[0.3,3.0], maxniter=50):
    """Fit the continuum using sigma clipping

    Args:
        x: The wavelengths
        y: The log-fluxes
        order: The polynomial order to use
        nsigma: The sigma clipping threshold: tuple (low, high)
        maxniter: The maximum number of iterations to do

    Returns:
        The value of the continuum at the wavelengths in x

    """
    A = np.vander(x - np.nanmean(x), order+1)
    m = np.ones(len(x), dtype=bool)
    for i in range(maxniter):
        m[ivars == 0] = 0  # mask out the bad pixels
        w = np.linalg.solve(np.dot(A[m].T, A[m]), np.dot(A[m].T, y[m]))
        mu = np.dot(A, w)
        resid = y - mu
        sigma = np.sqrt(np.nanmedian(resid**2))
        #m_new = np.abs(resid) < nsigma*sigma
        m_new = (resid > -nsigma[0]*sigma) & (resid < nsigma[1]*sigma)
        if m.sum() == m_new.sum():
            m = m_new
            break
        m = m_new
    return mu

def bin_data(xs, ys, ivars, xps):
    """
    Bin data onto a uniform grid using medians.
    
    Args:
        `xs`: `[N, M]` array of xs
        `ys`: `[N, M]` array of ys
        `ivars`: `[N, M]` array of y-ivars
        `xps`: `M'` grid of x-primes for output template
    
    Returns:
        `yps`: `M'` grid of y-primes
    
    """
    all_ys, all_xs, all_ivars = np.ravel(ys), np.ravel(xs), np.ravel(ivars)
    dx = xps[1] - xps[0] # ASSUMES UNIFORM GRID
    yps = np.zeros_like(xps)
    for i,t in enumerate(xps):
        ind = (all_xs >= t-dx/2.) & (all_xs < t+dx/2.)
        if np.sum(ind) > 0:
            #yps[i] = np.nanmedian(all_ys[ind])
            yps[i] = np.nansum(all_ys[ind] * all_ivars[ind]) / np.nansum(all_ivars[ind] + 1.) # MAGIC
    ind_nan = np.isnan(yps)
    yps.flat[ind_nan] = np.interp(xps[ind_nan], xps[~ind_nan], yps[~ind_nan])
    return xps, yps