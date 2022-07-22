Quickstart
==========

Installation
------------

*wobble* is currently under development and not yet available through pip. However, you can install the current developer version from GitHub:

.. code-block:: bash

	git clone https://github.com/megbedell/wobble.git
	cd wobble
	python setup.py develop
	
If you are running macOS 10.14 (Mojave) and find that setup fails at building the wobble C++ extensions, :doc:`try this <mojave>`.

You may also need to install some requirements, notably TensorFlow (which is best installed via pip rather than conda):

.. code-block:: bash

	pip install -r requirements.txt

If *wobble* builds with warnings and subsequently fails with a "Symbol not found" error on import, try rebuilding the C++ extensions using a different system compiler; see `this issue <https://github.com/megbedell/wobble/issues/66>`_.

Input Data
----------

Running *wobble* on a generic data set will require some initial processing.

If you just want some example data for testing purposes, try grabbing one of these pre-processed data sets: `51 Peg <https://www.dropbox.com/s/0jjdp5t3zto8hp7/51peg_e2ds.hdf5?dl=0>`_, `Barnard's Star <https://www.dropbox.com/s/ccd050p7g7vsjdq/barnards_e2ds.hdf5?dl=0>`_.

These data sets can be easily loaded as:

.. code-block:: python

	import wobble
	data = wobble.Data('filename.hdf5')

Assembling your own data set is possible via the built-in functions for HARPS, HARPS-N, HIRES, or ESPRESSO spectra, or by interactively passing pre-processed data. See the :doc:`API <api>` for full documentation of data loading options.

Running *wobble*
----------------

The following is an overview of the basic steps involved in running an analysis with *wobble*. If you want a simple example script, check `this demonstration Jupyter notebook <https://nbviewer.jupyter.org/github/megbedell/wobble/blob/master/notebooks/demo.ipynb>`_, designed to be used with the `51 Peg example data <https://www.dropbox.com/s/0jjdp5t3zto8hp7/51peg_e2ds.hdf5?dl=0>`_.

1. Load the data.

2. Create a wobble.Results object in which to store the calculation outputs.

3. Create a wobble.Model object.

.. note:: Data and Results objects span all echelle orders for the input spectra (or at least as many echelle orders as you specified in the *orders* keyword when loading Data), but Model is specific to a single order.

4. Populate the model with one or more spectral components. Model.add_star() and Model.add_telluric() are convenience functions for this purpose.

5. Optimize the model.

.. warning:: The optimization scheme in *wobble* depends on L1 and L2 regularization. The default values used by the model should work fine for most HARPS data, but if you are using a custom data set from a different instrument or if you see unexpected behavior in the spectral fits then the regularization amplitudes should be tuned before *wobble* can be run reliably. See `this Jupyter notebook <https://nbviewer.jupyter.org/github/megbedell/wobble/blob/master/notebooks/regularization.ipynb>`_ for an example of how to do this.

6. Repeat steps 3-5 as needed for additional echelle orders.

7. (Optional) Do post-processing within the results, including combining multiple orders to get a net RV time series; applying barycentric corrections; and applying instrumental drifts as available.

.. note:: Currently *wobble* does not support barycentric correction calculations; those should be supplied as a part of the input data. Also, the methods of combining orders and applying barycentric shifts may be sub-optimal.

8. Save the results and/or write out RVs for further use.

Here's an example of what that code might look like. Again, check out `the demo notebook <https://nbviewer.jupyter.org/github/megbedell/wobble/blob/master/notebooks/demo.ipynb>`_ for a slightly more detailed walkthrough, or :doc:`the API <api>` for advanced usage.

.. code-block:: python

	results = wobble.Results(data=data)
	for r in range(len(data.orders)): # loop through orders
		model = wobble.Model(data, results, r)
		model.add_star('star')
		model.add_telluric('tellurics')
		wobble.optimize_order(model)

	results.combine_orders('star') # post-processing: combine all orders
	results.apply_bervs('star') # post-processing: shift to barycentric frame
	results.apply_drifts('star') # post-processing: remove instrumental drift

	results.write_rvs('star', 'star_rvs.csv') # save just RVs
	results.write('results.hdf5') # save everything

Accessing *wobble* Outputs
--------------------------

All of the outputs from *wobble* are stored in the wobble.Results object. You can download example results files corresponding to the above data files here: `51 Peg <https://www.dropbox.com/s/em4irz97zxqopx4/results_51peg_Kstar0_Kt3.hdf5?dl=0>`_, `Barnard's Star <https://www.dropbox.com/s/ymcu2awo1v05rps/results_barnards_Kstar0_Kt0.hdf5?dl=0>`_.

A saved wobble.Results object can be loaded up from disk:

.. code-block:: python

	results = wobble.Results(filename='results.hdf5')
	print(results.component_names)
	
The names of the components are needed to access the associated attributes of each component. For example, let's say that two components are called 'star' and 'tellurics,' as in the example above. We can plot the mean templates for the two components in order `r` as follows:

.. code-block:: python

	import matplotlib.pyplot as plt
	plt.plot(np.exp(results.star_template_xs[r]), np.exp(results.star_template_ys[r]), 
			 label='star')
	plt.plot(np.exp(results.tellurics_template_xs[r]), np.exp(results.tellurics_template_ys[r]),
		 	label='tellurics')
	plt.xlabel('Wavelength (Ang)')
	plt.ylabel('Normalized Flux')
	plt.legend()
	plt.show()
	
And the RV time series can be plotted as follows:

.. code-block:: python

	plt.errorbar(results.dates, results.star_time_rvs, 
				 results.star_time_sigmas, 'k.')
	plt.xlabel('RV (m/s)')
	plt.ylabel('JD')
	plt.show()
	
Other useful quantities stored in the Results object include `results.ys_predicted`, which is an order R by epoch N by pixel M array of `y'` model predictions in the data space, and `results.[component name]_ys_predicted`, which is a same-sized array storing the contribution of a given component to the model prediction.

See the `demo Jupyter notebook <https://github.com/megbedell/wobble/blob/master/notebooks/demo.ipynb>`_ or the `notebook used to generate figures for the paper <https://github.com/megbedell/wobble/blob/master/paper/figures/make_figures.ipynb>`_ for further examples.