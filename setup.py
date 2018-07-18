#!/usr/bin/env python

import os
import tensorflow as tf
from setuptools import setup, Extension


include_dirs = [".", tf.sysconfig.get_include()]
include_dirs.append(os.path.join(
    include_dirs[1], "external/nsync/public"))

extensions = [
    Extension(
        "wobble.interp.interp_op",
        sources=[
            "wobble/interp/searchsorted_op.cc",
        ],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=["-std=c++11", "-stdlib=libc++"],
    ),
]

setup(
    name="wobble",
    version="0.0.1",
    license="MIT",
    author="Megan Bedell",
    author_email="mbedell@flatironinstitute.org",
    description="precise radial velocities with tellurics",
    packages=["wobble", "wobble.interp"],
    ext_modules=extensions,
    zip_safe=True,
)
