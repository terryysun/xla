/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/ir/replica_group.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/printer.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla {

std::string ReplicaGroupsToString(
    absl::Span<const ReplicaGroup> replica_groups) {
  std::vector<std::string> replica_group_str;
  replica_group_str.reserve(replica_groups.size());
  for (const ReplicaGroup& group : replica_groups) {
    replica_group_str.push_back(
        absl::StrCat("{", absl::StrJoin(group.replica_ids(), ","), "}"));
  }
  return absl::StrCat("{", absl::StrJoin(replica_group_str, ","), "}");
}

/************** MeshAxesReplicaGroupList implementation ***********************/

void HandleSingleAxisRefPerDimension(const AxisRef& axis,
                                     int64_t full_axis_size,
                                     std::vector<int64_t>& out_reshape_dims,
                                     std::vector<int64_t>& out_aggregate_axes) {
  if (axis.sub_axis_info().has_value()) {
    out_reshape_dims = {axis.sub_axis_info()->pre_size,
                        axis.sub_axis_info()->size,
                        full_axis_size / axis.sub_axis_info()->next_pre_size()};
    // The aggregation axis is the second dimension.
    out_aggregate_axes = {1};
  } else {
    out_reshape_dims = {full_axis_size};
    out_aggregate_axes = {0};
  }
}

void HandleMultiAxisRefPerDimension(std::vector<AxisRef>& axes,
                                    int64_t full_axis_size,
                                    std::vector<int64_t>& out_reshape_dims,
                                    std::vector<int64_t>& out_aggregate_axes) {
  // --- 1. Sort Axes and Original Indices Together ---
  // Sort both the axes and the original indices based on
  // sub_axis_info()->pre_size. This allows us to maintain user specified order
  // of AxisRef while still building the reshape and aggregate axes.
  std::vector<int> original_order(axes.size());
  std::iota(original_order.begin(), original_order.end(), 0);
  std::sort(original_order.begin(), original_order.end(),
            [&axes](int i, int j) {
              return axes[i].sub_axis_info()->pre_size <
                     axes[j].sub_axis_info()->pre_size;
            });
  std::sort(axes.begin(), axes.end(), [](const AxisRef& a, const AxisRef& b) {
    return a.sub_axis_info()->pre_size < b.sub_axis_info()->pre_size;
  });

  // --- 2. Build Reshape Dims and Aggregation Axes ---
  int64_t current_dim_index = 0;  // Index in the new reshaped tensor
  int64_t prefix_product = 1;     // Product of the size of all prior dimensions

  for (const AxisRef& axis : axes) {
    int64_t pre_size = axis.sub_axis_info()->pre_size;
    int64_t size = axis.sub_axis_info()->size;

    // Insert "padding" dimension if the current prefix product doesn't match
    // the required pre_size
    if (pre_size != prefix_product) {
      int64_t padding_size = pre_size / prefix_product;
      out_reshape_dims.push_back(padding_size);
      current_dim_index++;
      prefix_product *= padding_size;
    }

    // Insert the sharded size (the part to aggregate)
    out_reshape_dims.push_back(size);
    out_aggregate_axes.push_back(
        current_dim_index);  // This is the axis we aggregate over
    current_dim_index++;
    prefix_product *= size;
  }

  // Insert "suffix" dimension if the full size hasn't been reached
  if (prefix_product != full_axis_size) {
    out_reshape_dims.push_back(full_axis_size / prefix_product);
  }

  // --- 3. Permute Aggregate Axes back to Original Order ---
  // The aggregate axes were calculated based on the sorted list.
  // We must map them back to the original order to compute the correct
  // flattened replica groups.
  std::vector<int64_t> permuted_aggregate_axes(original_order.size());
  for (int64_t i = 0; i < original_order.size(); ++i) {
    permuted_aggregate_axes[original_order[i]] = out_aggregate_axes[i];
  }
  out_aggregate_axes = permuted_aggregate_axes;
}

bool ValidateSingleDimensionAxes(int64_t dim, std::vector<AxisRef>& axes,
                                 const Mesh& mesh_) {
  // If there's only one axis, nothing to check.
  if (axes.size() <= 1) {
    return true;
  }
  // --- Step 1: Deduplication ---
  // If one is a "full" axis (no sub_axis_info), it subsumes all other AxisRefs
  // with sub_axis_info.
  for (const AxisRef& axis : axes) {
    if (!axis.sub_axis_info().has_value()) {
      LOG(WARNING) << "MeshAxesReplicaGroupList: Redundant axis definition at "
                      "dimension: "
                   << dim
                   << ". Keeping only the full axis: " << axis.ToString(mesh_);
      axes = {axis};
      return true;
    }
  }
  // --- Step 2: Overlap Check ---
  // At this point, all remaining axes MUST have sub_axis_info().
  // Verify that the remaining multiple sub-axes do not overlap.
  for (int64_t i = 0; i < axes.size() - 1; ++i) {
    for (int64_t j = i + 1; j < axes.size(); ++j) {
      // CHECK will terminate the program on failure, matching original
      // behavior.
      CHECK(axes[i].CanCoexist(axes[j]))
          << "Overlapping sub-axes detected: " << axes[i].ToString(mesh_)
          << " and " << axes[j].ToString(mesh_);
    }
  }
  return true;  // Passed all checks for this dimension.
}

MeshAxesReplicaGroupList::MeshAxesReplicaGroupList(Mesh mesh,
                                                   std::vector<AxisRef> axes)
    : mesh_(std::move(mesh)), axes_(std::move(axes)) {
  if (num_devices_per_group() == 1) {
    LOG(ERROR) << "MeshAxesReplicaGroupList: " << ToString()
               << " has only one device per replica group.";
  }

  absl::flat_hash_set<int64_t> dimensions;
  absl::flat_hash_map<int64_t, std::vector<AxisRef>> dim_to_axes;
  for (const AxisRef& axis : axes_) {
    dim_to_axes[axis.mesh_axis_index()].push_back(axis);
    dimensions.insert(axis.mesh_axis_index());
    if (axis.sub_axis_info().has_value()) {
      CHECK(mesh_.axis_size(axis.mesh_axis_index()) %
                axis.sub_axis_info()->next_pre_size() ==
            0)
          << "Next pre-size must divide the full axis size.";
    }
  }

  // Validate input AxisRefs.
  for (int64_t dim : dimensions) {
    std::vector<AxisRef>& axes = dim_to_axes[dim];
    CHECK(ValidateSingleDimensionAxes(dim, axes, mesh_));
  }
}

int64_t MeshAxesReplicaGroupList::num_replica_groups() const {
  return mesh_.device_assignment().num_elements() / num_devices_per_group();
}

int64_t MeshAxesReplicaGroupList::num_devices_per_group() const {
  // Number of devices per replica group is equal to the product of the sizes of
  // all axes.
  int64_t devices_per_group = 1;
  for (const AxisRef& axis : axes_) {
    int64_t axis_size = axis.sub_axis_info().has_value()
                            ? axis.sub_axis_info()->size
                            : mesh_.axis_size(axis.mesh_axis_index());
    devices_per_group *= axis_size;
  }
  return devices_per_group;
}

std::vector<std::vector<int64_t>> get_replica_groups_for_full_axes(
    const Mesh& mesh, absl::Span<const int64_t> axis_sizes,
    const absl::Span<const int64_t> grouped_axes,
    const int64_t num_replica_groups, const int64_t num_devices_per_group) {
  // Reshape the device assignment array bases on the axis sizes and transpose
  // grouped axes to the end.
  std::vector<int> transpose_axes;
  transpose_axes.reserve(axis_sizes.size());
  for (int64_t i = 0; i < axis_sizes.size(); ++i) {
    if (!absl::c_linear_search(grouped_axes, i)) {
      transpose_axes.push_back(i);
    }
  }
  for (int64_t grouped_axis : grouped_axes) {
    transpose_axes.push_back(grouped_axis);
  }

  TileAssignment device_assignment =
      mesh.device_assignment().Reshape(axis_sizes).Transpose(transpose_axes);

  std::vector<std::vector<int64_t>> replica_groups;
  replica_groups.reserve(num_replica_groups);
  for (auto it = device_assignment.array().begin();
       it != device_assignment.array().end(); it += num_devices_per_group) {
    std::vector<int64_t> group(it, it + num_devices_per_group);
    replica_groups.emplace_back(std::move(group));
  }
  return replica_groups;
}

void MeshAxesReplicaGroupList::InitializeDimToReshapeAndAggregateAxes() {
  absl::flat_hash_map<int64_t, std::vector<AxisRef>> dim_to_axes;
  for (const AxisRef& axis : axes_) {
    dim_to_axes[axis.mesh_axis_index()].push_back(axis);
  }
  absl::flat_hash_map<int64_t, ReshapeAndAggregateAxes> dim_map;
  // For each dimension determine the reshape that is consistent with it's
  // AxisRef(s). Then maintain this reshape and the aggregated dims for easier
  // computation of replica groups. As an example for @mesh<"a"=8>
  // {a}               -> no reshape, aggregate over [0]
  // {a:(1)2}          -> reshape [8]->[1,2,4], aggregate over [1]
  // {a:(1)2, a:(4)2}  -> reshape [8]->[2,2,2], aggregate over [0,2]
  for (auto& [dim, axes] : dim_to_axes) {
    int64_t full_axis_size = mesh_.axis_size(dim);
    ReshapeAndAggregateAxes reshape_and_aggregate_axes;
    if (axes.size() == 1) {
      HandleSingleAxisRefPerDimension(
          axes[0], full_axis_size, reshape_and_aggregate_axes.reshape_dims,
          reshape_and_aggregate_axes.aggregate_axes);
    } else {
      // Otherwise dimension is a set of axes with sub-axes info.
      HandleMultiAxisRefPerDimension(axes, full_axis_size,
                                     reshape_and_aggregate_axes.reshape_dims,
                                     reshape_and_aggregate_axes.aggregate_axes);
    }
    dim_map[dim] = reshape_and_aggregate_axes;
  }
  dim_to_reshape_and_aggregate_axes_ = dim_map;
}

std::vector<std::vector<int64_t>>
MeshAxesReplicaGroupList::flattened_replica_groups() {
  if (!dim_to_reshape_and_aggregate_axes_.has_value()) {
    InitializeDimToReshapeAndAggregateAxes();
  }

  absl::flat_hash_map<int64_t, ReshapeAndAggregateAxes> dim_map =
      dim_to_reshape_and_aggregate_axes_.value();
  std::vector<int64_t> reindex_axis_sizes;
  std::vector<int64_t> reindexed_grouped_axes;
  for (int64_t i = 0; i < mesh_.axis_sizes().size(); ++i) {
    int64_t axis_size = mesh_.axis_size(i);
    auto it = dim_map.find(i);
    if (it == dim_map.end()) {
      reindex_axis_sizes.push_back(axis_size);
      continue;
    }
    int64_t offset_index = reindex_axis_sizes.size();
    const ReshapeAndAggregateAxes& reshape_and_aggregate_axes = it->second;
    for (int64_t reshape_dim : reshape_and_aggregate_axes.reshape_dims) {
      reindex_axis_sizes.push_back(reshape_dim);
    }
    for (int64_t aggregate_dim : reshape_and_aggregate_axes.aggregate_axes) {
      reindexed_grouped_axes.push_back(aggregate_dim + offset_index);
    }
  }
  return get_replica_groups_for_full_axes(
      mesh_, reindex_axis_sizes, reindexed_grouped_axes, num_replica_groups(),
      num_devices_per_group());
}

void MeshAxesReplicaGroupList::Print(Printer* printer) const {
  printer->Append(ToString());
}

std::string MeshAxesReplicaGroupList::ToString() const {
  std::string rg_str = "";
  // Add the axes defining the replica group, using names from the mesh.
  std::vector<std::string> group_axes_str;
  group_axes_str.reserve(axes_.size());
  for (const AxisRef& axis : axes_) {
    std::string axis_str = axis.ToString(mesh_);
    group_axes_str.push_back(axis_str);
  }
  absl::StrAppend(&rg_str, mesh_.ToString(), " {",
                  absl::StrJoin(group_axes_str, ","), "}");
  return rg_str;
}

MeshAxesReplicaGroupListProto MeshAxesReplicaGroupList::ToProto() const {
  MeshAxesReplicaGroupListProto proto;
  *proto.mutable_mesh() = mesh_.ToProto();
  for (const AxisRef& axis : axes_) {
    *proto.add_axes() = axis.ToProto();
  }
  return proto;
}

MeshAxesReplicaGroupList MeshAxesReplicaGroupList::FromProto(
    const MeshAxesReplicaGroupListProto& proto) {
  Mesh mesh = Mesh::FromProto(proto.mesh());
  std::vector<AxisRef> axes;
  for (const AxisRefProto& axis_proto : proto.axes()) {
    axes.push_back(AxisRef::FromProto(axis_proto));
  }
  return MeshAxesReplicaGroupList(mesh, axes);
}

/************** IotaReplicaGroupList implementation ***************************/
int64_t IotaReplicaGroupList::num_replica_groups() const {
  DCHECK_GE(num_replica_groups_, 0);
  return num_replica_groups_;
}

int64_t IotaReplicaGroupList::num_devices_per_group() const {
  DCHECK_GE(num_devices_per_group_, 0);
  return num_devices_per_group_;
}

std::string IotaReplicaGroupList::ToString() const {
  return iota_tile_assignment_.ToString();
}

void IotaReplicaGroupList::Print(Printer* printer) const {
  iota_tile_assignment_.Print(printer);
}

IotaReplicaGroupListProto IotaReplicaGroupList::ToProto() const {
  IotaReplicaGroupListProto proto;
  proto.set_num_replica_groups(num_replica_groups_);
  proto.set_num_devices_per_group(num_devices_per_group_);
  proto.mutable_iota_reshape_dims()->Assign(
      iota_tile_assignment_.reshape_dims().begin(),
      iota_tile_assignment_.reshape_dims().end());
  proto.mutable_iota_transpose_perm()->Assign(
      iota_tile_assignment_.transpose_perm().begin(),
      iota_tile_assignment_.transpose_perm().end());
  return proto;
}

IotaReplicaGroupList IotaReplicaGroupList::FromProto(
    const IotaReplicaGroupListProto& proto) {
  return IotaReplicaGroupList(
      proto.num_replica_groups(), proto.num_devices_per_group(),
      std::vector<int64_t>(proto.iota_reshape_dims().begin(),
                           proto.iota_reshape_dims().end()),
      std::vector<int>(proto.iota_transpose_perm().begin(),
                       proto.iota_transpose_perm().end()));
}

std::vector<std::vector<int64_t>>
IotaReplicaGroupList::flattened_replica_groups() const {
  std::vector<std::vector<int64_t>> result;
  result.reserve(num_replica_groups());
  Array<int64_t> array = ToArray();
  for (auto it = array.begin(); it != array.end();
       it += num_devices_per_group()) {
    result.emplace_back(it, it + num_devices_per_group());
  }
  return result;
}

namespace {
std::shared_ptr<std::vector<ReplicaGroup>> ExpandIota(
    const IotaReplicaGroupList& iota) {
  VLOG(3) << "Expanding iota replica group list: " << iota.ToString();
  auto result = std::make_shared<std::vector<ReplicaGroup>>();
  const int64_t num_replica_groups = iota.num_replica_groups();
  result->reserve(num_replica_groups);

  Array<int64_t> array = iota.ToArray();
  // Iota replica group list array must only have 2 dimensions.
  DCHECK_EQ(array.num_dimensions(), 2);
  const int64_t num_devices_per_group = iota.num_devices_per_group();
  DCHECK_EQ(array.end() - array.begin(),
            num_devices_per_group * num_replica_groups);
  for (auto it = array.begin(); it != array.end();
       it += num_devices_per_group) {
    auto& group = result->emplace_back();
    *group.mutable_replica_ids() = {it, it + num_devices_per_group};
  }
  return result;
}
}  // namespace

/************** CollectiveDeviceList implementation ***************************/
const std::vector<ReplicaGroup>& CollectiveDeviceList::replica_groups() const {
  if (replica_groups_ == nullptr) {
    CHECK(iota_replica_group_list_.has_value());
    replica_groups_ = ExpandIota(iota_replica_group_list_.value());
    CHECK(replica_groups_ != nullptr);
  }
  return *replica_groups_;
}

std::string CollectiveDeviceList::ToString(
    bool print_full_replica_group_list) const {
  if (iota_replica_group_list_.has_value() && !print_full_replica_group_list) {
    return iota_replica_group_list_->ToString();
  }

  return ReplicaGroupsToString(replica_groups());
}

void CollectiveDeviceList::Print(Printer* printer,
                                 bool print_full_replica_group_list) const {
  if (iota_replica_group_list_.has_value() && !print_full_replica_group_list) {
    iota_replica_group_list_->Print(printer);
    return;
  }
  printer->Append("{");
  bool leading_comma = false;
  for (const ReplicaGroup& group : replica_groups()) {
    printer->AppendInt64List(group.replica_ids(), leading_comma);
    leading_comma = true;
  }
  printer->Append("}");
}

CollectiveDeviceListProto CollectiveDeviceList::ToProto() const {
  CollectiveDeviceListProto proto;
  if (iota_replica_group_list_.has_value()) {
    *(proto.mutable_iota_replica_group_list()) =
        iota_replica_group_list_->ToProto();
    return proto;
  }

  proto.mutable_replica_groups()->Assign(replica_groups().begin(),
                                         replica_groups().end());
  return proto;
}

CollectiveDeviceList CollectiveDeviceList::FromProto(
    const CollectiveDeviceListProto& proto) {
  if (proto.has_iota_replica_group_list()) {
    return CollectiveDeviceList(
        IotaReplicaGroupList::FromProto(proto.iota_replica_group_list()));
  }

  if (proto.replica_groups_size() > 0) {
    return CollectiveDeviceList(proto.replica_groups().begin(),
                                proto.replica_groups().end());
  }

  return CollectiveDeviceList();
}

CollectiveDeviceList CollectiveDeviceList::FromProto(
    const HloInstructionProto& proto) {
  // Create CollectiveDeviceList from legacy field (replica_groups) if it is
  // populated.
  if (proto.replica_groups_size() > 0) {
    VLOG(10) << "Creating collective device list from proto using legacy "
                "replica groups field.";
    return CollectiveDeviceList(proto.replica_groups().begin(),
                                proto.replica_groups().end());
  }

  if (!proto.has_collective_device_list()) {
    return CollectiveDeviceList();
  }

  // Create CollectiveDeviceList from non-legacy field (collective_device_list).
  return FromProto(proto.collective_device_list());
}

}  // namespace xla
