#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("InterpGrad")
  .Attr("T: {float, double}")
  .Input("t: T")
  .Input("x: T")
  .Input("bf: T")
  .Output("by: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle x, y, t, bf;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &t));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &x));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &bf));

    TF_RETURN_IF_ERROR(c->Merge(t, bf, &bf));

    c->set_output(0, c->input(1));

    return Status::OK();
  });

template <typename T>
class InterpGradOp : public OpKernel {
 public:
  explicit InterpGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& t_tensor = context->input(0);
    const Tensor& x_tensor = context->input(1);
    const Tensor& bf_tensor = context->input(2);

    // Dimensions
    const int64 M = t_tensor.NumElements();
    const int64 N = x_tensor.NumElements();
    int64 m = M-1;
    OP_REQUIRES(context, (bf_tensor.NumElements() == M),
        errors::InvalidArgument("Dimension mismatch"));

    // Output
    Tensor* by_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_tensor.shape(), &by_tensor));

    // Access the data
    const auto t = t_tensor.template flat<T>();
    const auto x = x_tensor.template flat<T>();
    const auto bf = bf_tensor.template flat<T>();
    auto by = by_tensor->template flat<T>();

    for (int64 n = 0; n < N; ++n) by(n) = T(0.0);

    while ((m >= 0) && (t(m) > x(N-1))) {
      by(N-1) += bf(m);
      m--;
    }
    if (m < 0) return;

    for (int64 n = N-2; n >= 0; --n) {
      auto dx = x(n+1) - x(n);
      while (t(m) > x(n)) {
        auto s = bf(m) * (t(m) - x(n)) / dx;
        by(n) += bf(m) - s;
        by(n+1) += s;
        m--;
        if (m < 0) return;
      }
    }

    while (m >= 0) {
      by(0) += bf(m);
      m--;
    }

  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("InterpGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),         \
      InterpGradOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
