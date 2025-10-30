/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/buffers_nan_count_thunk.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_buffer_id.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/resource_requests.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

namespace se = stream_executor;

using ::stream_executor::gpu::BufferDebugLog;
using ::testing::UnorderedElementsAre;

class BuffersDebugNanCountThunkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_,
                            se::PlatformManager::PlatformWithName("CUDA"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
    allocator_ =
        std::make_unique<se::StreamExecutorMemoryAllocator>(stream_->parent());

    if (!executor_->GetDeviceDescription()
             .cuda_compute_capability()
             .IsAtLeastPascal()) {
      GTEST_SKIP()
          << "buffer nan-counting is not supported on CUDA architectures "
             "older than Pascal due to missing atomic fetch_add with "
             "system scope";
    }
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::StreamExecutorMemoryAllocator> allocator_;
};

TEST_F(BuffersDebugNanCountThunkTest, CalculatesNanCounts) {
  static constexpr size_t kLogSize = BufferDebugLog::RequiredSizeForEntries(10);
  static constexpr size_t kInputElems = 1024;
  static constexpr size_t kInputSizeInBytes = kInputElems * sizeof(float);
  static constexpr size_t kTotalDeviceMemoryBytes =
      kLogSize + kInputSizeInBytes * 2;
  // Setup memory allocations for the log and inputs
  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kTotalDeviceMemoryBytes,
                         /*color=*/0);
  BufferAllocation::Slice log_slice(&alloc, /*offset=*/0, kLogSize);
  BufferAllocation::Slice inputs[2];
  for (int i = 0; i < 2; ++i) {
    inputs[i] = BufferAllocation::Slice(
        &alloc, /*offset=*/kLogSize + i * kInputSizeInBytes, kInputSizeInBytes);
  }
  BufferAllocations allocations(
      {executor_->AllocateArray<uint8_t>(kTotalDeviceMemoryBytes)},
      executor_->device_ordinal(), allocator_.get());
  se::DeviceMemoryBase log_mem = allocations.GetDeviceAddress(log_slice);
  se::DeviceMemoryBase inputs0_mem = allocations.GetDeviceAddress(inputs[0]);
  se::DeviceMemoryBase inputs1_mem = allocations.GetDeviceAddress(inputs[1]);
  // Initialize the log in device memory
  TF_ASSERT_OK_AND_ASSIGN(BufferDebugLog device_log,
                          BufferDebugLog::CreateOnDevice(
                              *stream_, se::DeviceMemory<uint8_t>(log_mem)));
  // Fill inputs with some data
  std::vector<float> data(kInputElems, 0);
  data[123] = std::numeric_limits<float>::quiet_NaN();
  TF_ASSERT_OK(stream_->Memcpy(&inputs0_mem, data.data(), kInputSizeInBytes));
  data[123] = 0;
  data[456] = std::numeric_limits<float>::quiet_NaN();
  data[789] = std::numeric_limits<float>::quiet_NaN();
  TF_ASSERT_OK(stream_->Memcpy(&inputs1_mem, data.data(), kInputSizeInBytes));
  // Setup parameters for Initialize/Prepare/ExecuteOnStream
  Thunk::InitializeParams init_params;
  init_params.executor = executor_;
  init_params.stream = stream_.get();
  ResourceRequests resource_requests;
  auto execute_params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), allocations, stream_.get(),
      /*command_buffer_trace_stream=*/stream_.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr);

  BuffersDebugNanCountThunk thunk(
      Thunk::ThunkInfo(), log_slice,
      {{ThunkBufferId::Create(ThunkId(123), 4).value(),
        {inputs[0], PrimitiveType::F32}},
       {ThunkBufferId::Create(ThunkId(456), 8).value(),
        {inputs[1], PrimitiveType::F32}}});
  TF_ASSERT_OK(thunk.Initialize(init_params));
  TF_ASSERT_OK(thunk.Prepare(Thunk::PrepareParams{}, resource_requests));
  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<BufferDebugLogEntry> entries,
                          device_log.ReadFromDevice(*stream_));

  // BuffersDebugNanCountThunk launches a kernel for each input buffer, they may
  // complete in any order.
  EXPECT_THAT(
      entries,
      UnorderedElementsAre(
          BufferDebugLogEntry{
              /*entry_id=*/ThunkBufferId::Create(ThunkId(123), 4).value(),
              /*value=*/1,
          },
          BufferDebugLogEntry{
              /*entry_id=*/ThunkBufferId::Create(ThunkId(456), 8).value(),
              /*value=*/2,
          }));
}

}  // namespace
}  // namespace xla::gpu
