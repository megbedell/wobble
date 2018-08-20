# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["interp"]

import tensorflow as tf
from ..tf_utils import load_op_library

mod = load_op_library(__file__, "interp_op")
# searchsorted = mod.searchsorted


def interp(t, x, y, **kwargs):
    return mod.interp(t, x, y, **kwargs)[0]
#     x_ext = tf.concat((x[:1], x, x[-1:]), axis=0)
#     y_ext = tf.concat((y[:1], y, y[-1:]), axis=0)
#     dx = x_ext[1:] - x_ext[:-1]
#     dy = y_ext[1:] - y_ext[:-1]
#     dx = tf.where(tf.greater(tf.abs(dx), tf.zeros_like(dx)),
#                   dx, tf.ones_like(dx))

#     x0 = tf.gather(x_ext, inds)
#     y0 = tf.gather(y_ext, inds)
#     slope = tf.gather(dy / dx, inds)

#     return slope * (t - x0) + y0


@tf.RegisterGradient("Interp")
def _interp_grad(op, *grads):
    t, x, y = op.inputs
    v, inds = op.outputs
    bv = grads[0]
    bt, by = mod.interp_rev(t, x, y, inds, bv)
    return [bt, None, by]
