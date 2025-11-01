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

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_STABLEHLOLOWERTOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

class LowerTranspose : public mlir::OpRewritePattern<stablehlo::TransposeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::TransposeOp op,
      mlir::PatternRewriter& rewriter) const override {
    SmallVector<int32_t> permutation =
        llvm::to_vector_of<int32_t>(op.getPermutation());
    rewriter.replaceOpWithNewOp<ttir::TransOp>(op, op.getResult().getType(),
                                               op.getOperand(), permutation);
    return mlir::success();
  }
};

class LowerIotaToMakeRange : public mlir::OpRewritePattern<stablehlo::IotaOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::IotaOp op, mlir::PatternRewriter& rewriter) const override {
    auto result_type = op.getResult().getType();

    if (result_type.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "tt.make_range is only supported for 1D outputs.");
    }

    if (!result_type.getElementType().isInteger(32)) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "tt.make_range is only supported for integer types.");
    }

    if (result_type.getElementType().isUnsignedInteger(32)) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "lowering to tt.make_range is only supported for 32 bit signed "
          "integers.");
    }

    auto iota_end = result_type.getDimSize(0);

    rewriter.replaceOpWithNewOp<ttir::MakeRangeOp>(op, result_type,
                                                   /*start=*/0, iota_end);
    return mlir::success();
  }
};

class LowerBroadcastInDim
    : public mlir::OpRewritePattern<stablehlo::BroadcastInDimOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::BroadcastInDimOp op,
      mlir::PatternRewriter& rewriter) const override {
    ::xla::EmitterLocOpBuilder builder(op.getLoc(), rewriter);
    auto input_tensor = op.getOperand();
    auto input_shape = input_tensor.getType().getShape();
    auto output_shape = op.getResult().getType().getShape();
    auto broadcast_dims = op.getBroadcastDimensions();

    if (input_shape.empty()) {
      auto broadcast_dim_input = op.getOperand();
      auto broadcast_dim_input_element_type =
          broadcast_dim_input.getType().getElementType();

      auto extracted = mlir::tensor::ExtractOp::create(
          rewriter, op.getLoc(), broadcast_dim_input_element_type,
          broadcast_dim_input);

      rewriter.replaceOpWithNewOp<ttir::SplatOp>(op, op.getResult().getType(),
                                                 extracted);
      return mlir::success();
    }
    int64_t axis = 0;
    int64_t input_dim_id = 0;
    for (int output_dim_id = 0; output_dim_id < output_shape.size();
         output_dim_id++) {
      if (input_dim_id < broadcast_dims.size() &&
          output_dim_id == broadcast_dims[input_dim_id]) {
        // The dim is not broadcasted. Validate matching dim sizes.
        CHECK_EQ(input_shape[input_dim_id], output_shape[output_dim_id]);
        ++input_dim_id;
        axis = output_dim_id + 1;
        continue;
      }
      input_tensor = builder.create<ttir::ExpandDimsOp>(input_tensor, axis);
    }
    rewriter.replaceOpWithNewOp<ttir::BroadcastOp>(op, op.getResult().getType(),
                                                   input_tensor);

    return mlir::success();
  }
};

class LowerReduce : public mlir::OpRewritePattern<stablehlo::ReduceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::ReduceOp op, mlir::PatternRewriter& rewriter) const override {
    if (mlir::failed(VerifyOpIsCompatibleWithTritonReduce(op, rewriter))) {
      return mlir::failure();
    }

    int32_t axis = op.getDimensions()[0];

    // In case shlo returns a 0 rank tensor triton needs to return a scalar as
    // triton doesn't support 0 rank tensors.
    SmallVector<Type> adjusted_result_types;
    adjusted_result_types.reserve(op.getNumResults());
    for (auto result : op.getResults()) {
      auto shaped_type = cast<mlir::ShapedType>(result.getType());
      if (shaped_type.getRank() == 0) {
        adjusted_result_types.push_back(shaped_type.getElementType());
      } else {
        adjusted_result_types.push_back(shaped_type);
      }
    }

    auto triton_reduce_op = ttir::ReduceOp::create(
        rewriter, op.getLoc(), adjusted_result_types, op.getInputs(), axis);

    Region& triton_reduce_region = triton_reduce_op.getCombineOp();
    rewriter.cloneRegionBefore(op.getBody(), triton_reduce_region,
                               triton_reduce_region.end());
    Block& triton_reduce_region_block = triton_reduce_region.front();
    for (mlir::BlockArgument& argument :
         triton_reduce_region_block.getArguments()) {
      auto extract_op = cast<mlir::tensor::ExtractOp>(*argument.user_begin());

      auto scalar_type =
          cast<mlir::RankedTensorType>(argument.getType()).getElementType();
      argument.setType(scalar_type);
      rewriter.replaceOp(extract_op, argument);
    }

    Operation* terminator = triton_reduce_region_block.getTerminator();
    rewriter.setInsertionPointToEnd(&triton_reduce_region_block);
    SmallVector<Value> return_operands;
    for (Value operand : terminator->getOperands()) {
      auto defining_op = operand.getDefiningOp();
      return_operands.push_back(mlir::tensor::ExtractOp::create(
          rewriter, defining_op->getLoc(),
          cast<mlir::RankedTensorType>(operand.getType()).getElementType(),
          defining_op->getResult(0)));
    }
    rewriter.replaceOpWithNewOp<ttir::ReduceReturnOp>(terminator,
                                                      return_operands);

    rewriter.replaceOp(op, triton_reduce_op);

    // Replace usages of the original op results. If the original result was a
    // 0-rank tensor, we need to wrap the scalar result of tt.reduce in a
    // tensor.from_elements op.
    rewriter.setInsertionPointAfter(triton_reduce_op);
    for (const auto& triton_result : triton_reduce_op.getResults()) {
      if (mlir::isa<mlir::ShapedType>(triton_result.getType())) {
        continue;
      }
      auto extract_op =
          cast<mlir::tensor::ExtractOp>(*triton_result.user_begin());

      rewriter.replaceOp(extract_op, triton_result);
    }
    return mlir::success();
  }

  // Verifies that the stablehlo reduce op can be lowered to a triton reduce
  // op.
  // This checks that proper emitting of `tensor.from_elements` and
  // `tensor.extract` on reducer inputs and outputs has happened. It also checks
  // that `tensor.extract` was emitted on the result of the reduce operation if
  // the result is a zero rank tensor.
  mlir::LogicalResult VerifyOpIsCompatibleWithTritonReduce(
      stablehlo::ReduceOp op, mlir::PatternRewriter& rewriter) const {
    if (mlir::failed(VerifyReducerArgs(op, rewriter))) {
      return mlir::failure();
    }

    if (mlir::failed(VerifyReducerResults(op, rewriter))) {
      return mlir::failure();
    }

    if (mlir::failed(VerifyResults(op, rewriter))) {
      return mlir::failure();
    }

    // Check that the reduction is along a single dimension.
    auto dimensions = op.getDimensions();
    if (dimensions.size() != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "tt.reduce only supports single dimension reductions.");
    }

    return mlir::success();
  }

  mlir::LogicalResult VerifyReducerArgs(stablehlo::ReduceOp op,
                                        mlir::PatternRewriter& rewriter) const {
    // Check that all arguments get extracted into a scalar.
    for (mlir::BlockArgument& argument : op.getBody().front().getArguments()) {
      if (!argument.hasOneUse()) {
        return rewriter.notifyMatchFailure(
            op, "Expected a single user for an argument to a reduce combiner.");
      }
      if (!dyn_cast<mlir::tensor::ExtractOp>(*argument.user_begin())) {
        return rewriter.notifyMatchFailure(op,
                                           "Expected a tensor extract op as "
                                           "user of reduce combiner argument.");
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult VerifyReducerResults(
      stablehlo::ReduceOp op, mlir::PatternRewriter& rewriter) const {
    // Check that all outputs get created by a from_elements op.
    for (Value operand : op.getBody().front().getTerminator()->getOperands()) {
      if (!operand.hasOneUse()) {
        return rewriter.notifyMatchFailure(
            op->getLoc(),
            "Expected a single user for an output of a reduce combiner.");
      }
      auto from_elements =
          operand.getDefiningOp<mlir::tensor::FromElementsOp>();
      if (!from_elements) {
        return rewriter.notifyMatchFailure(op->getLoc(),
                                           "Expected a from_elements op as "
                                           "user of reduce combiner output.");
      }
    }
    return mlir::success();
  }

  mlir::LogicalResult VerifyResults(stablehlo::ReduceOp op,
                                    mlir::PatternRewriter& rewriter) const {
    // Check that all results get created by a from_elements op.
    for (Value result : op.getResults()) {
      auto shaped_type = cast<mlir::ShapedType>(result.getType());
      // If the result is a shaped type, then we don't need to do anything.
      if (shaped_type.getRank() != 0) {
        continue;
      }

      if (!result.hasOneUse()) {
        return rewriter.notifyMatchFailure(
            op->getLoc(), "Expected a single user for reduce result.");
      }

      auto extract_op = dyn_cast<mlir::tensor::ExtractOp>(*result.user_begin());

      if (!extract_op) {
        return rewriter.notifyMatchFailure(
            op->getLoc(),
            "Expected a tensor extract op as "
            "the only user of 0 rank reduce result.");
      }
    }
    return mlir::success();
  }
};

class LowerReshape : public mlir::OpRewritePattern<stablehlo::ReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::ReshapeOp op, mlir::PatternRewriter& rewriter) const override {
    // Conservatively prevent Triton from reordering elements within the tile.
    // TODO(b/353637689): see if this restriction can be lifted.
    bool allow_reorder = false;
    rewriter.replaceOpWithNewOp<ttir::ReshapeOp>(
        op, op.getResult().getType(), op.getOperand(), allow_reorder);
    return mlir::success();
  }
};

class StableHLOLowerToTritonPass
    : public impl::StableHLOLowerToTritonPassBase<StableHLOLowerToTritonPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<LowerTranspose, LowerIotaToMakeRange, LowerBroadcastInDim,
                 LowerReduce, LowerReshape>(mlir_context);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateStableHLOLowerToTritonPass() {
  return std::make_unique<StableHLOLowerToTritonPass>();
}

}  // namespace mlir::triton::xla
