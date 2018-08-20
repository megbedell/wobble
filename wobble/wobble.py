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




__all__ = ["optimize_order", "optimize_orders"]


            

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