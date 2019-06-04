import numpy as np
import h5py
import os
import wobble
from wobble.regularization import *

if __name__ == "__main__":
    # change these keywords:
    starname = '51peg'
    datafile = '../data/{0}_e2ds.hdf5'.format(starname)
    R = 72 # the number of echelle orders total in the data set
    orders = np.arange(72) # list of indices for the echelle orders to be tuned
    K_star = 0 # number of variable components for stellar spectrum
    K_t = 0 # number of variable components for telluric spectrum         
    tellurics_template_fixed = False    
    plot = True # warning: this will generate many plots!
    verbose = True # warning: this will print a lot of info & progress bars!
    
    # create directory for plots if it doesn't exist:
    if plot:
        plot_dir = '../regularization/{0}_Kstar{1}_Kt{2}/'.format(starname, K_star, K_t)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    
    # create regularization parameter files if they don't exist:
    R = len(orders)
    star_filename = '../wobble/regularization/{0}_star_K{1}.hdf5'.format(starname, K_star)
    if not os.path.isfile(star_filename):
        generate_regularization_file(star_filename, R, type='star')
    tellurics_filename = '../wobble/regularization/{0}_t_K{1}.hdf5'.format(starname, K_t)
    if not os.path.isfile(tellurics_filename):                
        generate_regularization_file(tellurics_filename, R, type='telluric')
        
    # load up the data we'll use for training:
    data = wobble.Data(datafile, orders=orders) # to get N_epochs    

    # choose validation epochs:
    validation_epochs = np.random.choice(data.N, data.N//8, replace=False) # 12.5% of epochs will be validation set
        
    # improve each order's regularization:
    for r,o in enumerate(orders):
        if verbose:
            print('---- STARTING ORDER {0} ----'.format(o))
            print("starting values:")
            print("star:")
            with h5py.File(star_filename, 'r') as f:
                for key in list(f.keys()):
                    print("{0}: {1:.0e}".format(key, f[key][o]))
            print("tellurics:")
            with h5py.File(tellurics_filename, 'r') as f:
                for key in list(f.keys()):
                    print("{0}: {1:.0e}".format(key, f[key][o]))
        objs = setup_for_order(r, data, validation_epochs)
        improve_order_regularization(o, star_filename, tellurics_filename,
                                         *objs,
                                         verbose=verbose, plot=plot, 
                                         basename='{0}o{1}'.format(plot_dir, o), 
                                         K_star=K_star, K_t=K_t, L1=True, L2=True)
        if verbose:                                 
            print('---- ORDER {0} COMPLETE ({1}/{2}) ----'.format(o,r,len(orders)-1))
            print("best values:")
            print("star:")
            with h5py.File(star_filename, 'r') as f:
                for key in list(f.keys()):
                    print("{0}: {1:.0e}".format(key, f[key][o]))
            print("tellurics:")
            with h5py.File(tellurics_filename, 'r') as f:
                for key in list(f.keys()):
                    print("{0}: {1:.0e}".format(key, f[key][o]))        

    # save some extra summary plots:
    if plot:
        plot_pars_from_file(star_filename, 'regularization/{0}_star_Kstar{1}_Kt{2}'.format(starname, K_star, K_t), orders=orders)
        plot_pars_from_file(tellurics_filename, 'regularization/{0}_tellurics_Kstar{1}_Kt{2}'.format(starname, K_star, K_t), orders=orders)  