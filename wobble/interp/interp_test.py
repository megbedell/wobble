#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator

from . import interp


class InterpTest(tf.test.TestCase):

    def get_vals_1d(self):
        np.random.seed(1234)
        x = np.sort(np.random.uniform(0, 10, 50))
        y = np.sin(x)
        t = np.random.uniform(x.min(), x.max(), 10)
        return t, x, y

    def get_vals_2d_t(self):
        np.random.seed(1234)
        x = np.sort(np.random.uniform(0, 10, 50))
        y = np.sin(x)
        t = np.random.uniform(x.min(), x.max(), (10, 5))
        return t, x, y

    def get_vals_2d_txy(self):
        np.random.seed(1234)
        x = np.sort(np.random.uniform(0, 10, (5, 50)), axis=1)
        y = np.sin(x)
        t = np.random.uniform(x.min(), x.max(), (y.shape[0], 10))
        return t, x, y

    def get_vals_2d_ty(self):
        np.random.seed(1234)
        x = np.sort(np.random.uniform(0, 10, 50))
        y = np.vstack([np.sin(x + i) for i in range(5)])
        t = np.random.uniform(x.min(), x.max(), (y.shape[0], 10))
        return t, x, y

    def get_vals_2d_tx(self):
        np.random.seed(1234)
        x = np.sort(np.random.uniform(0, 10, (5, 50)), axis=1)
        y = np.sum(np.sin(x), axis=0)
        t = np.random.uniform(x.min(), x.max(), (x.shape[0], 10))
        return t, x, y

    def test_value_1d(self):
        with self.cached_session():
            t, x, y = self.get_vals_1d()
            res = interp(t, x, y)
            sp_res = RegularGridInterpolator([x], y)(t)
            assert np.allclose(res.eval(), sp_res)

    def test_value_2d_t(self):
        with self.cached_session():
            t, x, y = self.get_vals_2d_t()
            res = interp(t, x, y)
            sp_res = RegularGridInterpolator([x], y)(t.flatten())
            sp_res = sp_res.reshape(t.shape)
            assert np.allclose(res.eval(), sp_res)

    def test_value_2d_txy(self):
        with self.cached_session():
            t, x, y = self.get_vals_2d_txy()
            res = interp(t, x, y)
            sp_res = []
            for i in range(y.shape[0]):
                sp_res.append(RegularGridInterpolator([x[i]], y[i])(t[i]))
            sp_res = np.array(sp_res)
            assert np.allclose(res.eval(), sp_res)

    def test_value_2d_ty(self):
        with self.cached_session():
            t, x, y = self.get_vals_2d_ty()
            res = interp(t, x, y)
            sp_res = []
            for i in range(y.shape[0]):
                sp_res.append(RegularGridInterpolator([x], y[i])(t[i]))
            sp_res = np.array(sp_res)
            assert np.allclose(res.eval(), sp_res)

    def test_value_2d_tx(self):
        with self.cached_session():
            t, x, y = self.get_vals_2d_tx()
            res = interp(t, x, y)
            sp_res = []
            for i in range(x.shape[0]):
                sp_res.append(RegularGridInterpolator([x[i]], y)(t[i]))
            sp_res = np.array(sp_res)
            assert np.allclose(res.eval(), sp_res)

    def check_gradient(self, t, x, y):
        with self.cached_session():
            ti = tf.constant(t, dtype=tf.float64)
            yi = tf.constant(y, dtype=tf.float64)
            res = interp(ti, x, yi)
            err = tf.test.compute_gradient_error(
                [ti, yi], [t.shape, y.shape], res, res.eval().shape, [t, y],
                1e-8)
            assert np.allclose(err, 0.0, atol=1e-6)

    def test_gradient_1d(self):
        self.check_gradient(*self.get_vals_1d())

    def test_gradient_2d_t(self):
        self.check_gradient(*self.get_vals_2d_t())

    def test_gradient_2d_txy(self):
        self.check_gradient(*self.get_vals_2d_txy())

    def test_gradient_2d_tx(self):
        self.check_gradient(*self.get_vals_2d_tx())

    def test_gradient_2d_ty(self):
        self.check_gradient(*self.get_vals_2d_ty())


if __name__ == "__main__":
    tf.test.main()
