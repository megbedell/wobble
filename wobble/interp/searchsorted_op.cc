#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Searchsorted")
  .Attr("T: {float, double}")
  .Input("a: T")
  .Input("v: T")
  .Output("inds: int64")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle a, v;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &v));
    c->set_output(0, c->input(1));
    return Status::OK();
  });

template <typename T>
class SearchsortedOp : public OpKernel {
 public:
  explicit SearchsortedOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& a_tensor = context->input(0);
    const Tensor& v_tensor = context->input(1);

    // Dimensions
    int64 m = 0;
    const int64 N = a_tensor.NumElements();
    const int64 M = v_tensor.NumElements();

    // Output
    Tensor* inds_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_tensor.shape(), &inds_tensor));

    // Access the data
    const auto a = a_tensor.template flat<T>();
    const auto v = v_tensor.template flat<T>();
    auto inds = inds_tensor->flat<int64>();

    while ((m < M) && (v(m) <= a(0))) {
      inds(m) = 0;
      m++;
    }
    if (m >= M) return;

    for (int64 n = 0; n < N-1; ++n) {
      while (v(m) <= a(n+1)) {
        inds(m) = n+1;
        m++;
        if (m >= M) return;
      }
    }

    while (m < M) {
      inds(m) = N;
      m++;
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Searchsorted").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      SearchsortedOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
