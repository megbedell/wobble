import wobble
import numpy as np
import matplotlib.pyplot as plt
import copy

if __name__ == "__main__":
    data1 = wobble.Data('51peg_e2ds.hdf5', filepath='../data/', orders=[53])
    data2 = wobble.Data('barnards_e2ds.hdf5', filepath='../data/', orders=[53])
    assert len(data1.orders) == len(data2.orders)
    data = copy.deepcopy(data1)
    for attr in ['dates', 'bervs', 'pipeline_rvs', 'pipeline_sigmas', 'airms', 'drifts', 'filelist', 'origin_file']:
        setattr(data, attr, np.append(getattr(data1, attr), getattr(data2, attr)))
    for attr in ['fluxes', 'xs', 'flux_ivars', 'ivars', 'ys']:
        data1_attr = getattr(data1, attr)
        data2_attr = getattr(data2, attr)
        data_attr = [np.append(data1_attr[i], data2_attr[i], axis=0) for i in range(data.R)]
        setattr(data, attr, data_attr)
    data.epoch_groups = [data1.epochs, data1.N + data2.epochs]
    data.N = data1.N + data2.N
    
    results = wobble.Results(data=data)
    r = 0
    model = wobble.Model(data, results, r)   

    model.add_star('star1', variable_bases=0, epochs=data.epoch_groups[0], 
                    regularization_par_file='../wobble/regularization/51peg_star_K0.hdf5')
    model.add_star('star2', variable_bases=0, epochs=data.epoch_groups[1], 
                    regularization_par_file='../wobble/regularization/barnards_star_K0.hdf5')
    model.add_telluric('tellurics', variable_bases=0, epochs=None, 
                        regularization_par_file='../wobble/regularization/51peg_t_K0.hdf5')
                
    wobble.optimize_order(model, niter=150, uncertainties=False)

    plt.plot(np.exp(results.star1_template_xs[r]), np.exp(results.star1_template_ys[r]), label='star1', alpha=0.8)
    plt.plot(np.exp(results.star2_template_xs[r]), np.exp(results.star2_template_ys[r]), label='star2', alpha=0.8)
    plt.plot(np.exp(results.tellurics_template_xs[r]), np.exp(results.tellurics_template_ys[r]), label='tellurics')
    plt.plot(np.exp(data2.xs[0][0] + np.log(wobble.doppler(results.star2_rvs[0][data1.N], tensors=False))), 
             np.exp(data2.ys[0][0]), 'k.', alpha=0.7)
    plt.legend(fontsize=14)
    plt.xlim([5728,5733])
    #plt.ylim([0.95,1.02])
    plt.ylim([0.5,1.1])
    plt.show()

