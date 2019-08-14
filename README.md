## wobble *(/wäb.lā/)* <a href='http://astro.uchicago.edu/~bmontet/wobble.mp3'>:sound:</a>

[![Documentation Status](https://readthedocs.org/projects/wobble/badge/?version=latest)](https://wobble.readthedocs.io/en/latest/?badge=latest)[![arXiv](https://img.shields.io/badge/arXiv-1901.00503-orange.svg)](https://arxiv.org/abs/1901.00503)

*wobble* is an open-source python package for analyzing time-series spectra. It was designed with stabilized extreme precision radial velocity (EPRV) spectrographs in mind, but is highly flexible and extensible to a variety of applications. It takes a data-driven approach to deriving radial velocities and requires no *a priori* knowledge of the stellar spectrum or telluric features.

*wobble* is under development.

To install the current developer version:

```bash
git clone https://github.com/megbedell/wobble.git
cd wobble
python setup.py develop
```

The paper presenting the *wobble* method used a slightly older version of the code than what is currently in the master branch. To access this version, install *wobble* and then change branches:
```
git checkout paper-version
```
