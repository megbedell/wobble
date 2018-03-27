import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import wobble

if __name__ == "__main__":
    data = wobble.Data('hip54287_e2ds.hdf5', filepath='data/', orders=np.arange(72))
    print("data loaded")
    
    model = wobble.Model(data)
    model.add_star('hip54287b')
    model.add_telluric('tellurics')
    print(model)
    
    plot_dir = 'results/plots/'
    for r in range(data.R):
        niter = 80
        nll_history, rvs_history, model_history, chis_history = wobble.optimize_order(model, data, r, 
                niter=niter, output_history=True) 
        plt.scatter(np.arange(len(nll_history)), nll_history)
        ax = plt.gca()
        ax.set_yscale('log')
        plt.savefig(plot_dir+'nll_order{0}.png'.format(r))   
        plt.clf()
        rvs_ani = wobble.plot_rv_history(data, rvs_history, niter, 50, xlims=[-20000, 20000], ylims=None)
        rvs_ani.save(plot_dir+'rvs_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
        print('RVs animation saved')
        session = wobble.get_session()
        model_xs = session.run(model.components[0].model_xs[r])
        model_ani = wobble.plot_model_history(model_xs, model_history, niter, 50)
        model_ani.save(plot_dir+'model_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
        print('model animation saved')
        data_xs = session.run(data.xs[r])
        chis_ani = wobble.plot_chis_history(0, data_xs, chis_history, niter, 50)
        chis_ani.save(plot_dir+'chis_order{0}_epoch0.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
        print('chis animation saved')