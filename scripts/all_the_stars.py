import wobble
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = wobble.Data('51peg_e2ds.hdf5', filepath='../data/', orders=[56])
    results = wobble.Results(data=data)
    r = 0
    model = wobble.Model(data, results, r)
    epochs1 = np.random.choice(np.arange(data.N), 20, replace=False)
    epochs2 = np.arange(data.N)[~np.isin(np.arange(data.N), epochs1)]
    model.add_star('star1', variable_bases=0, epochs=epochs1, 
                    regularization_par_file='../wobble/regularization/51peg_star_K0.hdf5')
    model.add_star('star2', variable_bases=0, epochs=epochs2, 
                    regularization_par_file='../wobble/regularization/51peg_star_K0.hdf5')
    model.add_telluric('tellurics', variable_bases=0, epochs=None, 
                        regularization_par_file='../wobble/regularization/51peg_t_K3.hdf5')
                
    wobble.optimize_order(model, niter=100, uncertainties=False)

    plt.plot(np.exp(results.star1_template_xs[r]), np.exp(results.star1_template_ys[r]), label='star1', alpha=0.8)
    plt.plot(np.exp(results.star2_template_xs[r]), np.exp(results.star2_template_ys[r]), label='star2', alpha=0.8)
    plt.plot(np.exp(results.tellurics_template_xs[r]), np.exp(results.tellurics_template_ys[r]), label='tellurics')
    plt.legend(fontsize=14)
    plt.show()

