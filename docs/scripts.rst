Helper Scripts
==============

A few python scripts are included in the *scripts* subdirectory. These scripts are not part of the core *wobble* functionality but may be helpful as guides for doing a few common operations related to *wobble*.

Making the Data
---------------

The `make_data.py script <https://github.com/megbedell/wobble/blob/master/scripts/make_data.py>`_ is intended to reformat HARPS data into an HDF5 file that is readable by *wobble*.

As an example, let's say that you have a set of HARPS spectra and their corresponding calibration data saved in the directory `/path/to/files`. You could write a custom script to reformat these data as follows:

.. code-block:: python

	from .make_data import read_data_from_fits, write_data, dimensions
	import glob
	filelist = glob.glob('/path/to/files/HARPS*ccf_*_A.fits') # list all star CCF files
	data = read_data_from_fits(filelist)
	write_data(*data, filelist, '../data/filename.hdf5')
	
Now the data are formatted, saved, and ready to be loaded with:

.. code-block:: python

	import wobble
	data = wobble.Data('filename.hdf5')
	
If you've downloaded these data from the `ESO archive 
<http://archive.eso.org/wdb/wdb/adp/phase3_main/form>`_, you may not have the wavelength calibration files. A helper function is provided to create a list of all missing wavelength files, which you can then download from the `HARPS calibration archive <http://archive.eso.org/wdb/wdb/eso/repro/form>`_:

.. code-block:: python

	from .make_data import missing_wavelength_files
	missing = missing_wavelength_files(filelist)
	print(missing)
	
All wavelength calibration files must be in place for the data to be reformatted successfully.
   
Tuning Regularization Parameters
--------------------------------

Regularization is an important part of `wobble`'s functionality. To get the best possible results for any given data set, the regularization strengths should be tuned to suit these data. We do this with a cross-validation scheme.

The `regularization.py script <https://github.com/megbedell/wobble/blob/master/scripts/regularization.py>`_ provides a framework for executing this cross-validation and saving the results in a *wobble*-friendly format.

Running *wobble*
----------------

Once the data have been processed and the regularization amplitudes have been set, running *wobble* should be a straightforward procedure. An overview can be found in the Quickstart section. As further examples, the scripts used to run the three analyses performed in the paper can be found under `script_51peg.py <https://github.com/megbedell/wobble/blob/master/scripts/script_51peg.py>`_, `script_barnards.py <https://github.com/megbedell/wobble/blob/master/scripts/script_barnards.py>`_, and `script_HD189733.py <https://github.com/megbedell/wobble/blob/master/scripts/script_HD189733.py>`_.