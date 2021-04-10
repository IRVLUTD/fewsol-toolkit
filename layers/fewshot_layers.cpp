#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/************************************************************
 backproject depth to 3D points
*************************************************************/

std::vector<at::Tensor> backproject_cuda_forward(
    float fx, float fy, float px, float py,
    at::Tensor depth);

std::vector<at::Tensor> backproject_forward(
    float fx, float fy, float px, float py,
    at::Tensor depth)
{
  CHECK_INPUT(depth);

  return backproject_cuda_forward(fx, fy, px, py, depth);
}


/********* python interface ***********/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backproject_forward", &backproject_forward, "backproject forward (CUDA)");
}
