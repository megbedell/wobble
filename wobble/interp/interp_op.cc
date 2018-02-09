#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Interp")
  .Attr("T: {float, double}")
  .Input("t: T")
  .Input("x: T")
  .Input("y: T")
  .Output("f: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle x, y, t;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &t));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &x));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &y));

    TF_RETURN_IF_ERROR(c->Merge(x, y, &y));

    c->set_output(0, c->input(0));

    return Status::OK();
  });

template <typename T>
class InterpOp : public OpKernel {
 public:
  explicit InterpOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& t_tensor = context->input(0);
    const Tensor& x_tensor = context->input(1);
    const Tensor& y_tensor = context->input(2);

    // Dimensions
    int64 m = 0;
    const int64 M = t_tensor.NumElements();
    const int64 N = x_tensor.NumElements();
    OP_REQUIRES(context, (y_tensor.NumElements() == N),
        errors::InvalidArgument("Dimension mismatch"));

    // Output
    Tensor* f_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, t_tensor.shape(), &f_tensor));

    // Access the data
    const auto t = t_tensor.template flat<T>();
    const auto x = x_tensor.template flat<T>();
    const auto y = y_tensor.template flat<T>();
    auto f = f_tensor->template flat<T>();

    while ((m < M) && (t(m) <= x(0))) {
      f(m) = y(0);
      m++;
    }
    if (m >= M) return;

    for (int64 n = 0; n < N-1; ++n) {
      auto dx = x(n+1) - x(n);
      auto dy = y(n+1) - y(n);
      while (t(m) <= x(n+1)) {
        f(m) = y(n) + dy * (t(m) - x(n)) / dx;
        m++;
        if (m >= M) return;
      }
    }

    while (m < M) {
      f(m) = y(N-1);
      m++;
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Interp").Device(DEVICE_CPU).TypeConstraint<type>("T"),         \
      InterpOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
