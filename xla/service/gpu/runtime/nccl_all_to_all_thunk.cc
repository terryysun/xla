/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/nccl_all_to_all_thunk.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "mlir/IR/Value.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

NcclAllToAllConfig GetNcclAllToAllConfig(const HloAllToAllInstruction* instr) {
  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfig(instr, std::nullopt);
  config.has_split_dimension = instr->split_dimension().has_value();
  return config;
}

}  // namespace

NcclAllToAllStartThunk::NcclAllToAllStartThunk(
    ThunkInfo thunk_info, NcclApi* nccl_api,
    const HloAllToAllInstruction* instr,
    std::vector<NcclCollectiveThunk::Buffer> buffers, bool p2p_memcpy_enabled)
    : NcclCollectiveThunk(Thunk::kNcclAllToAllStart, thunk_info, nccl_api,
                          IsSyncCollective(instr)),
      config_(GetNcclAllToAllConfig(instr)),
      buffers_(std::move(buffers)),
      p2p_memcpy_enabled_(p2p_memcpy_enabled) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ absl::Status NcclAllToAllStartThunk::CheckImplementable(
    const HloAllToAllInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  auto status = [&instr]() -> absl::Status {
    std::optional<uint64_t> split_dim = instr->split_dimension();
    for (HloInstruction* operand : instr->operands()) {
      Shape shape = operand->shape();
      TF_RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kNcclAllToAll));
      if (split_dim &&
          !ShapeUtil::IsEffectivelyMostMajorDimension(shape, *split_dim)) {
        return absl::UnimplementedError(absl::Substitute(
            "all-to-all split dim $0 is not the most major in input shape $1",
            *split_dim, shape.ToString(/*print_layout=*/true)));
      }
    }
    return absl::OkStatus();
  };
  return AddOpDescription<NcclAllToAllStartThunk>(
      status(), instr, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclAllToAllStartThunk::GetGroupMode(
    const HloAllToAllInstruction* instr) {
  return GetNcclAllToAllConfig(instr).config.group_mode;
}

absl::Status NcclAllToAllStartThunk::Initialize(
    const InitializeParams& params) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::Initialize(params));
  device_count_ = params.local_device_count;
  CHECK_GT(device_count_, 0);
  VLOG(5) << "Local device count: " << device_count_;
  return absl::OkStatus();
}

absl::Status NcclAllToAllStartThunk::Cleanup(
    const CleanupParams& params) {
  if (p2p_memcpy_enabled_ && receive_pointer_map_) {
    if (!params.executor->HostMemoryUnregister(receive_pointer_map_)) {
      VLOG(5) << "Unregistering host receive pointer for memcpy failed.";
    }
  }
  return absl::OkStatus();
}

absl::Status NcclAllToAllStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    NcclCommHandleWrapper comm_wrapper) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(const int64_t current_id,
                      GetCurrentId(params.collective_params, config_));
  TF_ASSIGN_OR_RETURN(int32_t num_participants,
                      nccl_api()->CommCount(comm_wrapper.comm_handle));
  if (!receive_pointer_map_) {
    if (!params.stream->parent()->HostMemoryRegister(
            receive_pointer_map_,
            sizeof(void*) * num_participants * num_participants)) {
        VLOG(5) << "Registering host receive pointer for memcpy failed.";
      }
  }

  if (is_local() && p2p_memcpy_enabled_) {
    return xla::gpu::RunMemCpyAllToAll(
        nccl_api(), config_.has_split_dimension, device_buffers, stream,
        comm_wrapper.comm_handle, current_id, receive_pointer_map_);
  }
  return xla::gpu::RunAllToAll(nccl_api(), config_.has_split_dimension,
                               device_buffers, stream,
                               comm_wrapper.comm_handle);
}

AsyncStreamKind NcclAllToAllStartThunk::GetAsyncStreamKind() const {
  return (is_local() && p2p_memcpy_enabled_) ? AsyncStreamKind::kMemCpyP2P
                                             : AsyncStreamKind::kCollective;
}

bool NcclAllToAllStartThunk::is_local() const {
  for (const auto& replica_group : config_.config.replica_groups) {
    const int64_t node_id = replica_group.replica_ids().at(0) / device_count_;
    if (!absl::c_all_of(replica_group.replica_ids(),
                        [this, node_id](const int64_t rank) {
                          return rank / device_count_ == node_id;
                        })) {
      return false;
    }
  }
  return true;
}

absl::Status RunAllToAll(NcclApi* nccl_api, bool has_split_dimension,
                         std::vector<DeviceBufferPair>& buffers,
                         se::Stream& stream, NcclApi::NcclCommHandle comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-to-all from device ordinal: " << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(nccl_api, device_ordinal, buffers, comm));

  TF_ASSIGN_OR_RETURN(int32_t num_participants, nccl_api->CommCount(comm));

  TF_RETURN_IF_ERROR(nccl_api->GroupStart());

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (has_split_dimension) {
    for (DeviceBufferPair& buffer : buffers) {
      TF_RET_CHECK(buffer.element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";

      size_t chunk_elements = buffer.element_count / num_participants;

      for (int peer = 0; peer < num_participants; ++peer) {
        se::DeviceMemoryBase send_slice =
            NcclApi::Slice(buffer.source_buffer, buffer.element_type,
                           peer * chunk_elements, chunk_elements);

        se::DeviceMemoryBase recv_slice =
            NcclApi::Slice(buffer.destination_buffer, buffer.element_type,
                           peer * chunk_elements, chunk_elements);

        TF_RETURN_IF_ERROR(nccl_api->Send(send_slice, buffer.element_type,
                                          chunk_elements, peer, comm, &stream));

        TF_RETURN_IF_ERROR(nccl_api->Recv(recv_slice, buffer.element_type,
                                          chunk_elements, peer, comm, &stream));
      }
    }
  } else {
    TF_RET_CHECK(buffers.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];

      TF_RETURN_IF_ERROR(
          nccl_api->Send(buffer.source_buffer, buffer.element_type,
                         buffer.element_count, i, comm, &stream));

      TF_RETURN_IF_ERROR(
          nccl_api->Recv(buffer.destination_buffer, buffer.element_type,
                         buffer.element_count, i, comm, &stream));
    }
  }

  return nccl_api->GroupEnd();
}

absl::Status RunMemCpyAllToAll(NcclApi* nccl_api, bool has_split_dimension,
                               std::vector<DeviceBufferPair>& buffers,
                               se::Stream& stream, NcclApi::NcclCommHandle comm,
                               int64_t current_id, void** recv_ptr_map) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing mem-copy-all-to-all from device ordinal: "
          << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(nccl_api, device_ordinal, buffers, comm));

  TF_ASSIGN_OR_RETURN(int32_t num_participants, nccl_api->CommCount(comm));

  TF_RETURN_IF_ERROR(nccl_api->GroupStart());

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (has_split_dimension) {
    for (DeviceBufferPair& buffer : buffers) {
      TF_RET_CHECK(buffer.element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";

      size_t chunk_elements = buffer.element_count / num_participants;

      for (int peer = 0; peer < num_participants; ++peer) {
        se::DeviceMemoryBase recv_slice =
            NcclApi::Slice(buffer.destination_buffer, buffer.element_type,
                           peer * chunk_elements, chunk_elements);
        recv_ptr_map[current_id * num_participants + peer] =
            recv_slice.opaque();
      }
      nccl_api->AllGatherPtrs(&recv_ptr_map[current_id * num_participants],
                              &recv_ptr_map, num_participants, comm, &stream);
      VLOG(3) << "Using memcpy, received target pointer map: " << recv_ptr_map
              << " current_id " << current_id;

      for (int peer = 0; peer < num_participants; ++peer) {
        se::DeviceMemoryBase send_slice =
            NcclApi::Slice(buffer.source_buffer, buffer.element_type,
                           peer * chunk_elements, chunk_elements);
        se::DeviceMemoryBase dst_addr = se::DeviceMemoryBase(
            recv_ptr_map[peer * num_participants + current_id]);
        TF_RETURN_IF_ERROR(
            stream.MemcpyD2D(&dst_addr, send_slice, send_slice.size()));
      }
    }
  } else {
    TF_RET_CHECK(buffers.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    for (int peer = 0; peer < num_participants; ++peer) {
      recv_ptr_map[current_id * num_participants + peer] =
          buffers[peer].destination_buffer.opaque();
    }

    nccl_api->AllGatherPtrs(&recv_ptr_map[current_id * num_participants],
                            &recv_ptr_map, num_participants, comm, &stream);
    VLOG(3) << "Using memcpy, received target pointer map: " << recv_ptr_map
            << " current_id " << current_id;

    for (size_t peer = 0; peer < num_participants; ++peer) {
      // double buffer, exchange data with peer
      se::DeviceMemoryBase dst_addr = se::DeviceMemoryBase(
          recv_ptr_map[peer * num_participants + current_id]);
      TF_RETURN_IF_ERROR(stream.MemcpyD2D(&dst_addr,
                                          buffers[peer].source_buffer,
                                          buffers[peer].source_buffer.size()));
    }
  }

  return nccl_api->GroupEnd();
}

}  // namespace gpu
}  // namespace xla
