Helper Scripts
==============

A few python scripts are included in the *scripts* subdirectory. These scripts are not part of the core *wobble* functionality but may be helpful as guides for doing a few common operations related to *wobble*.
   
Tuning Regularization Parameters
--------------------------------

Regularization is an important part of `wobble`'s functionality. To get the best possible results for any given data set, the regularization strengths should be tuned to suit these data. We do this with a cross-validation scheme.

The `tune_regularization.py script <https://github.com/megbedell/wobble/blob/master/scripts/tune_regularization.py>`_ shows an example of generating new regularization parameter files and tuning them for a given data set via cross-validation.

More details are given under the API.

Running *wobble*
----------------

Once the data have been processed and the regularization amplitudes have been set, running *wobble* should be a straightforward procedure. An overview can be found in the Quickstart section. As further examples, the scripts used to run the three analyses performed in the paper can be found under `script_51peg.py <https://github.com/megbedell/wobble/blob/master/scripts/script_51peg.py>`_, `script_barnards.py <https://github.com/megbedell/wobble/blob/master/scripts/script_barnards.py>`_, and `script_HD189733.py <https://github.com/megbedell/wobble/blob/master/scripts/script_HD189733.py>`_.