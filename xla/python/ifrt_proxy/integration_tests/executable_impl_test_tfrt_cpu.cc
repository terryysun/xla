// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/test_util.h"

int main(int argc, char** argv) {
  const std::string disabled[] = {
      // Executable::IsDeleted always returns false with TFRT CPU backend.
      "LoadedExecutableImplTest.IsDeleted",

      // Enable this when Serialization support for IFRT IR is available.
      "IfrtIrExecutableImplTest.CallXla",
      "IfrtIrExecutableImplTest.CopyArrays",
      "IfrtIrExecutableImplTest.Reshard",
      "IfrtIrExecutableImplTest.ZeroInput",
      "IfrtIrExecutableImplTest.ZeroOutput",
      "IfrtIrExecutableImplTest.BufferDonation",
      "IfrtIrExecutableImplTest.LoadedExecBinding",
      "ProgramLoadedExecutableImplTest.MultipleAtomProgramsNeedDummyInputs",
  };

  const std::string filter = absl::StrCat("-", absl::StrJoin(disabled, ":"));

#ifdef GTEST_FLAG_SET
  GTEST_FLAG_SET(filter, filter.c_str());
#else
  testing::GTEST_FLAG(filter) = filter.c_str();
#endif

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
