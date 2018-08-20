import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from scipy.optimize import minimize
from tqdm import tqdm
import sys
import h5py
import copy
import pickle
import tensorflow as tf
T = tf.float64
import pdb

from .data import Data
from .results import Results
from .model import Model, Component
from .history import History

speed_of_light = 2.99792458e8   # m/s

__all__ = ["get_session", "doppler", "optimize_order", "optimize_orders"]

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
            

def optimize_order(model, *args, **kwargs):
    '''
    optimize the model for order r in data
    '''      
    model.setup()    
    model.optimize(*args, **kwargs)

def optimize_orders(data, **kwargs):
    """
    optimize model for all orders in data
    """
    for r in range(data.R):
        model = Model(data)
        print("--- ORDER {0} ---".format(r))
        if r == 0: 
            results = Results(data=data)
        optimize_order(model, results, **kwargs)
        #if (r % 5) == 0:
        #    results.write('results_order{0}.hdf5'.format(data.orders[r]))
    results.compute_final_rvs(model) 
    #results.write('results.hdf5')   
    return results    