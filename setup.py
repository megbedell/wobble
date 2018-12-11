#!/usr/bin/env python

import os
import sys
from setuptools import setup, Extension

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    
    # Skip installing the C extension on ReadTheDocs
    extensions = []
    packages = ["wobble"]
    
else:
    
    import tensorflow as tf
    compile_flags = tf.sysconfig.get_compile_flags()
    link_flags = tf.sysconfig.get_link_flags()
    
    compile_flags += ["-std=c++11"]
    if sys.platform == "darwin":
        compile_flags += ["-mmacosx-version-min=10.9"]

    extensions = [
        Extension(
            "wobble.interp.interp_op",
            sources=[
                "wobble/interp/interp_op.cc",
                "wobble/interp/interp_rev_op.cc",
            ],
            include_dirs=["wobble/interp"],
            language="c++",
            extra_compile_args=compile_flags,
            extra_link_args=link_flags,
        ),
    ]
    
    packages = ["wobble", "wobble.interp"]

setup(
    name="wobble",
    version="0.0.1",
    license="MIT",
    author="Megan Bedell",
    author_email="mbedell@flatironinstitute.org",
    description="precise radial velocities with tellurics",
    packages=packages,
    ext_modules=extensions,
    zip_safe=True,
)
