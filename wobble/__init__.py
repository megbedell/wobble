name = "wobble"
from .utils import bin_data, get_session, doppler, fit_continuum
from .interp import interp
from .data import Data, Spectrum
from .results import Results
from .model import Model, Component
from .history import History
from .regularization import *

def optimize_order(model, **kwargs):
    '''
    optimize the model for order r in data
    '''      
    model.setup()    
    h = model.optimize(**kwargs)
    if 'save_history' in kwargs and kwargs['save_history']:
        return h

def optimize_orders(data, **kwargs):
    """
    optimize model for all orders in data
    """
    results = Results(data=data)
    for r in range(data.R):
        model = Model(data, results, r)
        model.add_star('star')
        model.add_telluric('tellurics', rvs_fixed=True, variable_bases=3)
        print("--- ORDER {0} ---".format(r))
        optimize_order(model, **kwargs)
        if (r % 5) == 0:
            results.write('results_order{0}.hdf5'.format(data.orders[r]))
    results.compute_final_rvs(model) 
    results.write('results.hdf5')   
    return results 
