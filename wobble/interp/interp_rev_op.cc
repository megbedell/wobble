#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

using namespace tensorflow;

REGISTER_OP("InterpRev")
  .Attr("T: {float, double}")
  .Input("t: T")
  .Input("x: T")
  .Input("y: T")
  .Input("inds: int32")
  .Input("bv: T")
  .Output("bt: T")
  .Output("by: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle t, x, y, inds, bv;
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &t));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &x));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &y));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 1, &inds));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(4), 1, &bv));

    TF_RETURN_IF_ERROR(c->Merge(t, inds, &t));
    TF_RETURN_IF_ERROR(c->Merge(t, bv, &t));

    shape_inference::DimensionHandle dx = c->Dim(x, -1), dy = c->Dim(y, -1);
    TF_RETURN_IF_ERROR(c->Merge(dx, dy, &dx));

    c->set_output(0, c->input(0));
    c->set_output(1, c->input(2));
    return Status::OK();
  });

template <typename T>
class InterpRevOp : public OpKernel {
 public:
  explicit InterpRevOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& t_tensor    = context->input(0);
    const Tensor& x_tensor    = context->input(1);
    const Tensor& y_tensor    = context->input(2);
    const Tensor& inds_tensor = context->input(3);
    const Tensor& bv_tensor   = context->input(4);

    // Check that the dimensions are consistent
    int64 ndim = t_tensor.dims();
    OP_REQUIRES(context, ndim >= 1,
        errors::InvalidArgument("t must be at least 1D"));
    OP_REQUIRES(context, t_tensor.shape() == inds_tensor.shape(),
        errors::InvalidArgument("t and inds must be the same shape"));
    OP_REQUIRES(context, t_tensor.shape() == bv_tensor.shape(),
        errors::InvalidArgument("t and bv must be the same shape"));

    bool x_one_d = x_tensor.dims() == 1;
    bool y_one_d = y_tensor.dims() == 1;
    OP_REQUIRES(context, (x_tensor.dims() == ndim) || x_one_d,
        errors::InvalidArgument("x and t must have the same number of dimensions or x must be 1D"));
    OP_REQUIRES(context, (y_tensor.dims() == ndim) || y_one_d,
        errors::InvalidArgument("y and t must have the same number of dimensions or y must be 1D"));

    bool outer = y_tensor.dim_size(y_tensor.dims() - 1) == x_tensor.dim_size(x_tensor.dims() - 1);
    OP_REQUIRES(context, ((x_one_d || y_one_d) && outer) || (x_tensor.shape() == y_tensor.shape()),
        errors::InvalidArgument("x and y must be the same outer dimension"));

    // Compute the full size of the inner dimensions
    int64 size = 1;
    for (int64 k = 0; k < ndim - 1; ++k) {
      int64 dim = t_tensor.dim_size(k);
      size *= dim;
      OP_REQUIRES(context, x_one_d || (x_tensor.dim_size(k) == dim),
          errors::InvalidArgument("incompatible dimensions"));
      OP_REQUIRES(context, y_one_d || (y_tensor.dim_size(k) == dim),
          errors::InvalidArgument("incompatible dimensions"));
    }

    // The outer dimensions
    const int64 N = t_tensor.dim_size(ndim - 1);
    const int64 M = x_tensor.dim_size(x_tensor.dims() - 1);
    OP_REQUIRES(context, N <= tensorflow::kint32max,
        errors::InvalidArgument("too many elements in tensor"));
    OP_REQUIRES(context, M <= tensorflow::kint32max,
        errors::InvalidArgument("too many elements in tensor"));

    // Output
    Tensor* bt_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, t_tensor.shape(), &bt_tensor));
    Tensor* by_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, y_tensor.shape(), &by_tensor));

    // Access the data
    const auto t    = t_tensor.template flat_inner_dims<T, 2>();
    const auto x    = x_tensor.template flat_inner_dims<T, 2>();
    const auto y    = y_tensor.template flat_inner_dims<T, 2>();
    const auto inds = inds_tensor.flat_inner_dims<int, 2>();
    const auto bv   = bv_tensor.template flat_inner_dims<T, 2>();
    auto bt         = bt_tensor->template flat_inner_dims<T, 2>();
    auto by         = by_tensor->template flat_inner_dims<T, 2>();

    bt.setConstant(0.0);
    by.setConstant(0.0);

    for (int64 k = 0; k < size; ++k) {
      int64 kx = (x_one_d) ? 0 : k;
      int64 ky = (y_one_d) ? 0 : k;
      for (int64 n = 0; n < N; ++n) {
        T value = t(k, n);
        int ind = inds(k, n);
        if (ind <= 0) {
          by(ky, 0) += bv(k, n);
        } else if (ind >= M) {
          by(ky, M-1) += bv(k, n);
        } else {
          T factor = 1.0 / (x(kx, ind) - x(kx, ind-1));
          T a = (value - x(kx, ind-1)) * factor;
          T bvalue = bv(k, n) * (y(ky, ind) - y(ky, ind-1)) * factor;
          bt(k, n)     += bvalue;
          by(ky, ind)   += bv(k, n) * a;
          by(ky, ind-1) += bv(k, n) * (1.0 - a);
        }
      }
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("InterpRev").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      InterpRevOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
