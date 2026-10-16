#pragma once

#include "mln/sdk/functional/batchnormgrad.hpp"
#include "mln/sdk/functional/batchnorminference.hpp"
#include "mln/sdk/functional/batchnormtraining.hpp"
#include "mln/sdk/functional/cholesky.hpp"
#include "mln/sdk/functional/dynamicbroadcast.hpp"
#include "mln/sdk/functional/dynamicpad.hpp"
#include "mln/sdk/functional/dynamicreshape.hpp"
#include "mln/sdk/functional/einsum.hpp"
#include "mln/sdk/functional/getdimensionsize.hpp"
#include "mln/sdk/functional/optimizationbarrier.hpp"
#include "mln/sdk/functional/pad.hpp"
#include "mln/sdk/functional/realdynamicslice.hpp"
#include "mln/sdk/functional/reduceprecision.hpp"
#include "mln/sdk/functional/setdimensionsize.hpp"
#include "mln/sdk/functional/triangularsolve.hpp"
#include "mln/sdk/functional/unaryeinsum.hpp"
#include "syn/dynamic_shape.hpp"
#include "syn/operation.hpp"
#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    inline mlir::RankedTensorType result_type(Operations &ctx, mlir::OpBuilder &builder,
                                              mlir::Location loc, const Mln::Sdk::Tensor &input,
                                              const std::vector<int64_t> &shape) {
        mlir::Value operand = ctx.resolve(input, builder, loc);
        mlir::Type element = mlir::cast<mlir::ShapedType>(operand.getType()).getElementType();
        return mlir::RankedTensorType::get(mlir_shape(shape), element);
    }

    class GetDimensionSize : public Operation::For<Mln::Sdk::Functional::GetDimensionSize> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::GetDimensionSize &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                return mlir::stablehlo::GetDimensionSizeOp::create(
                           builder, loc, ctx.resolve(node.x, builder, loc), node.dimension)
                    .getResult();
            }
    };

    class SetDimensionSize : public Operation::For<Mln::Sdk::Functional::SetDimensionSize> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::SetDimensionSize &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                return mlir::stablehlo::SetDimensionSizeOp::create(
                           builder, loc, ctx.resolve(node.x, builder, loc),
                           ctx.resolve(node.size, builder, loc), node.dimension)
                    .getResult();
            }
    };

    class OptimizationBarrier : public Operation::For<Mln::Sdk::Functional::OptimizationBarrier> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::OptimizationBarrier &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                auto barrier = mlir::stablehlo::OptimizationBarrierOp::create(
                    builder, loc, ctx.resolve(node.x, builder, loc));
                return barrier.getResults().front();
            }
    };

    class ReducePrecision : public Operation::For<Mln::Sdk::Functional::ReducePrecision> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::ReducePrecision &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                return mlir::stablehlo::ReducePrecisionOp::create(
                           builder, loc, ctx.resolve(node.x, builder, loc), node.exponent_bits,
                           node.mantissa_bits)
                    .getResult();
            }
    };

    class Pad : public Operation::For<Mln::Sdk::Functional::Pad> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Pad &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                return mlir::stablehlo::PadOp::create(builder, loc,
                                                      ctx.resolve(node.x, builder, loc),
                                                      ctx.resolve(node.value, builder, loc),
                                                      node.low, node.high, node.interior)
                    .getResult();
            }
    };

    class DynamicReshape : public Operation::For<Mln::Sdk::Functional::DynamicReshape> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::DynamicReshape &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                mlir::Value input = ctx.resolve(node.x, builder, loc);
                mlir::Value shape = ctx.resolve(node.output_shape, builder, loc);
                return mlir::stablehlo::DynamicReshapeOp::create(
                           builder, loc, result_type(ctx, builder, loc, node.x, node.shape), input,
                           shape)
                    .getResult();
            }
    };

    class DynamicBroadcast : public Operation::For<Mln::Sdk::Functional::DynamicBroadcast> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::DynamicBroadcast &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                mlir::Value input = ctx.resolve(node.x, builder, loc);
                mlir::Value shape = ctx.resolve(node.output_dimensions, builder, loc);
                return mlir::stablehlo::DynamicBroadcastInDimOp::create(
                           builder, loc, result_type(ctx, builder, loc, node.x, node.shape), input,
                           shape, builder.getDenseI64ArrayAttr(node.dimensions))
                    .getResult();
            }
    };

    class RealDynamicSlice : public Operation::For<Mln::Sdk::Functional::RealDynamicSlice> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::RealDynamicSlice &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                return mlir::stablehlo::RealDynamicSliceOp::create(
                           builder, loc, result_type(ctx, builder, loc, node.x, node.shape),
                           ctx.resolve(node.x, builder, loc),
                           ctx.resolve(node.starts, builder, loc),
                           ctx.resolve(node.limits, builder, loc),
                           ctx.resolve(node.strides, builder, loc))
                    .getResult();
            }
    };

    class DynamicPad : public Operation::For<Mln::Sdk::Functional::DynamicPad> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::DynamicPad &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                return mlir::stablehlo::DynamicPadOp::create(
                           builder, loc, result_type(ctx, builder, loc, node.x, node.shape),
                           ctx.resolve(node.x, builder, loc), ctx.resolve(node.value, builder, loc),
                           ctx.resolve(node.low, builder, loc),
                           ctx.resolve(node.high, builder, loc),
                           ctx.resolve(node.interior, builder, loc))
                    .getResult();
            }
    };

    class Cholesky : public Operation::For<Mln::Sdk::Functional::Cholesky> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Cholesky &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                return mlir::stablehlo::CholeskyOp::create(
                           builder, loc, ctx.resolve(node.x, builder, loc), node.lower)
                    .getResult();
            }
    };

    class TriangularSolve : public Operation::For<Mln::Sdk::Functional::TriangularSolve> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::TriangularSolve &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                mlir::stablehlo::Transpose transpose = mlir::stablehlo::Transpose::NO_TRANSPOSE;
                if (node.transpose_a == Mln::Sdk::Functional::SolveTranspose::Transpose)
                    transpose = mlir::stablehlo::Transpose::TRANSPOSE;
                else if (node.transpose_a == Mln::Sdk::Functional::SolveTranspose::Adjoint)
                    transpose = mlir::stablehlo::Transpose::ADJOINT;
                return mlir::stablehlo::TriangularSolveOp::create(
                           builder, loc, ctx.resolve(node.a, builder, loc),
                           ctx.resolve(node.b, builder, loc), node.left_side, node.lower,
                           node.unit_diagonal, transpose)
                    .getResult();
            }
    };

    class BatchNormInference : public Operation::For<Mln::Sdk::Functional::BatchNormInference> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::BatchNormInference &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                mlir::Value input = ctx.resolve(node.x, builder, loc);
                return mlir::stablehlo::BatchNormInferenceOp::create(
                           builder, loc, input.getType(), input,
                           ctx.resolve(node.scale, builder, loc),
                           ctx.resolve(node.offset, builder, loc),
                           ctx.resolve(node.mean, builder, loc),
                           ctx.resolve(node.variance, builder, loc),
                           builder.getF32FloatAttr(node.epsilon),
                           builder.getI64IntegerAttr(node.feature_index))
                    .getResult();
            }
    };

    class BatchNormTraining : public Operation::MultiFor<Mln::Sdk::Functional::BatchNormTraining> {
        public:
            std::vector<mlir::Value> emit_typed(const Mln::Sdk::Functional::BatchNormTraining &node,
                                                Operations &ctx, mlir::OpBuilder &builder,
                                                mlir::Location loc) const override {
                mlir::Value input = ctx.resolve(node.x, builder, loc);
                auto operation = mlir::stablehlo::BatchNormTrainingOp::create(
                    builder, loc, input, ctx.resolve(node.scale, builder, loc),
                    ctx.resolve(node.offset, builder, loc), builder.getF32FloatAttr(node.epsilon),
                    builder.getI64IntegerAttr(node.feature_index));
                return {operation.getOutput(), operation.getBatchMean(), operation.getBatchVar()};
            }
    };

    class BatchNormGrad : public Operation::MultiFor<Mln::Sdk::Functional::BatchNormGrad> {
        public:
            std::vector<mlir::Value> emit_typed(const Mln::Sdk::Functional::BatchNormGrad &node,
                                                Operations &ctx, mlir::OpBuilder &builder,
                                                mlir::Location loc) const override {
                auto operation = mlir::stablehlo::BatchNormGradOp::create(
                    builder, loc, ctx.resolve(node.x, builder, loc),
                    ctx.resolve(node.scale, builder, loc), ctx.resolve(node.mean, builder, loc),
                    ctx.resolve(node.variance, builder, loc),
                    ctx.resolve(node.grad_output, builder, loc),
                    builder.getF32FloatAttr(node.epsilon),
                    builder.getI64IntegerAttr(node.feature_index));
                return {operation.getGradOperand(), operation.getGradScale(),
                        operation.getGradOffset()};
            }
    };

    class Einsum : public Operation::For<Mln::Sdk::Functional::Einsum> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Einsum &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                return mlir::stablehlo::EinsumOp::create(
                           builder, loc, result_type(ctx, builder, loc, node.lhs, node.shape),
                           ctx.resolve(node.lhs, builder, loc), ctx.resolve(node.rhs, builder, loc),
                           node.equation)
                    .getResult();
            }
    };

    class UnaryEinsum : public Operation::For<Mln::Sdk::Functional::UnaryEinsum> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::UnaryEinsum &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                return mlir::stablehlo::UnaryEinsumOp::create(
                           builder, loc, result_type(ctx, builder, loc, node.x, node.shape),
                           ctx.resolve(node.x, builder, loc), node.equation)
                    .getResult();
            }
    };
}
