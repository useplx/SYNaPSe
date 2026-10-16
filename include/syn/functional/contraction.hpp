#pragma once

#include "mln/sdk/functional/dot.hpp"
#include "mln/sdk/functional/dotgeneral.hpp"
#include "mln/sdk/functional/dynamiciota.hpp"
#include "syn/dynamic_shape.hpp"
#include "syn/operation.hpp"
#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    class DynamicIota : public Operation::For<Mln::Sdk::Functional::DynamicIota> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::DynamicIota &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                auto type = mlir::RankedTensorType::get(
                    mlir_shape(node.shape), dtype_to_mlir_type(node.dtype, *builder.getContext()));
                return mlir::stablehlo::DynamicIotaOp::create(
                           builder, loc, type, ctx.resolve(node.output_shape, builder, loc),
                           node.dimension)
                    .getResult();
            }
    };

    class Dot : public Operation::For<Mln::Sdk::Functional::Dot> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Dot &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                mlir::Value lhs = ctx.resolve(node.lhs, builder, loc);
                auto element = mlir::cast<mlir::ShapedType>(lhs.getType()).getElementType();
                auto type = mlir::RankedTensorType::get(mlir_shape(node.shape), element);
                return mlir::stablehlo::DotOp::create(builder, loc, type, lhs,
                                                      ctx.resolve(node.rhs, builder, loc), nullptr)
                    .getResult();
            }
    };

    class DotGeneral : public Operation::For<Mln::Sdk::Functional::DotGeneral> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::DotGeneral &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                mlir::Value lhs = ctx.resolve(node.lhs, builder, loc);
                auto element = mlir::cast<mlir::ShapedType>(lhs.getType()).getElementType();
                auto type = mlir::RankedTensorType::get(mlir_shape(node.shape), element);
                auto dimensions = mlir::stablehlo::DotDimensionNumbersAttr::get(
                    builder.getContext(), node.lhs_batch, node.rhs_batch, node.lhs_contract,
                    node.rhs_contract);
                return mlir::stablehlo::DotGeneralOp::create(builder, loc, type, lhs,
                                                             ctx.resolve(node.rhs, builder, loc),
                                                             dimensions, nullptr, nullptr)
                    .getResult();
            }
    };
}
