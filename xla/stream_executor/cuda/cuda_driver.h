/* Copyright 2015 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// CUDA userspace driver library wrapper functionality.

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_driver.h"

namespace stream_executor::gpu {

// CudaContext implements the Context class for CUDA GPUs.
class CudaContext : public Context {
 public:
  CudaContext(CUcontext context, int device_ordinal)
      : context_(context), device_ordinal_(device_ordinal) {}

  void SetActive() override;
  bool IsActive() const override;
  CUcontext context() const { return context_; }
  int device_ordinal() const override { return device_ordinal_; }
  absl::Status Synchronize() override;

  // Disallow copying and moving.
  CudaContext(CudaContext&&) = delete;
  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(CudaContext&&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

 private:
  CUcontext const context_;
  const int device_ordinal_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
