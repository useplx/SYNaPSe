#pragma once

#include "mln/sdk/functional/reduce.hpp"
#include "mln/sdk/functional/rng.hpp"
#include "mln/sdk/functional/rngbitgenerator.hpp"
#include "mln/sdk/functional/scatter.hpp"
#include "mln/sdk/functional/sort.hpp"
#include "syn/dynamic_shape.hpp"
#include "syn/operation.hpp"
#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    class Reduce : public Operation::For<Mln::Sdk::Functional::Reduce> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Reduce &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                mlir::Value input = ctx.resolve(node.x, builder, loc);
                mlir::Value initial = ctx.resolve(node.initial, builder, loc);
                auto element = mlir::cast<mlir::ShapedType>(input.getType()).getElementType();
                auto scalar = mlir::RankedTensorType::get({}, element);
                auto output = mlir::RankedTensorType::get(mlir_shape(node.shape), element);
                auto operation = mlir::stablehlo::ReduceOp::create(
                    builder, loc, mlir::TypeRange{output}, mlir::ValueRange{input},
                    mlir::ValueRange{initial},
                    mlir::DenseI64ArrayAttr::get(builder.getContext(), node.axes));

                mlir::OpBuilder::InsertionGuard guard(builder);
                mlir::Block *block =
                    builder.createBlock(&operation.getBody(), {}, {scalar, scalar}, {loc, loc});
                builder.setInsertionPointToStart(block);
                mlir::Value lhs = block->getArgument(0);
                mlir::Value rhs = block->getArgument(1);
                mlir::Value result;
                switch (node.reduction) {
                case Mln::Sdk::Functional::Reduction::Add:
                    result = mlir::stablehlo::AddOp::create(builder, loc, lhs, rhs);
                    break;
                case Mln::Sdk::Functional::Reduction::Multiply:
                    result = mlir::stablehlo::MulOp::create(builder, loc, lhs, rhs);
                    break;
                case Mln::Sdk::Functional::Reduction::Maximum:
                    result = mlir::stablehlo::MaxOp::create(builder, loc, lhs, rhs);
                    break;
                case Mln::Sdk::Functional::Reduction::Minimum:
                    result = mlir::stablehlo::MinOp::create(builder, loc, lhs, rhs);
                    break;
                case Mln::Sdk::Functional::Reduction::And:
                    result = mlir::stablehlo::AndOp::create(builder, loc, lhs, rhs);
                    break;
                case Mln::Sdk::Functional::Reduction::Or:
                    result = mlir::stablehlo::OrOp::create(builder, loc, lhs, rhs);
                    break;
                case Mln::Sdk::Functional::Reduction::Xor:
                    result = mlir::stablehlo::XorOp::create(builder, loc, lhs, rhs);
                    break;
                }
                mlir::stablehlo::ReturnOp::create(builder, loc, result);
                return operation.getResult(0);
            }
    };

    class Sort : public Operation::MultiFor<Mln::Sdk::Functional::Sort> {
        public:
            std::vector<mlir::Value> emit_typed(const Mln::Sdk::Functional::Sort &node,
                                                Operations &ctx, mlir::OpBuilder &builder,
                                                mlir::Location loc) const override {
                std::vector<mlir::Value> inputs;
                inputs.reserve(node.inputs.size());
                for (const auto &input : node.inputs)
                    inputs.push_back(ctx.resolve(input, builder, loc));

                auto operation = mlir::stablehlo::SortOp::create(
                    builder, loc, mlir::ValueRange(inputs), node.dimension, node.stable);
                std::vector<mlir::Type> scalar_types;
                scalar_types.reserve(inputs.size() * 2);
                for (mlir::Value input : inputs) {
                    auto element = mlir::cast<mlir::ShapedType>(input.getType()).getElementType();
                    auto scalar = mlir::RankedTensorType::get({}, element);
                    scalar_types.push_back(scalar);
                    scalar_types.push_back(scalar);
                }
                std::vector<mlir::Location> locations(scalar_types.size(), loc);
                mlir::OpBuilder::InsertionGuard guard(builder);
                mlir::Block *block =
                    builder.createBlock(&operation.getComparator(), {}, scalar_types, locations);
                builder.setInsertionPointToStart(block);
                auto direction = node.direction == Mln::Sdk::Functional::SortDirection::Ascending
                                     ? mlir::stablehlo::ComparisonDirection::LT
                                     : mlir::stablehlo::ComparisonDirection::GT;
                mlir::Value predicate = mlir::stablehlo::CompareOp::create(
                    builder, loc, block->getArgument(0), block->getArgument(1), direction);
                mlir::stablehlo::ReturnOp::create(builder, loc, predicate);

                return {operation.getResults().begin(), operation.getResults().end()};
            }
    };

    class Rng : public Operation::For<Mln::Sdk::Functional::Rng> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Rng &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                mlir::Value a = ctx.resolve(node.a, builder, loc);
                auto element = mlir::cast<mlir::ShapedType>(a.getType()).getElementType();
                auto type = mlir::RankedTensorType::get(mlir_shape(node.shape), element);
                auto distribution = node.distribution == Mln::Sdk::Functional::Distribution::Uniform
                                        ? mlir::stablehlo::RngDistribution::UNIFORM
                                        : mlir::stablehlo::RngDistribution::NORMAL;
                return mlir::stablehlo::RngOp::create(
                           builder, loc, type, a, ctx.resolve(node.b, builder, loc),
                           ctx.resolve(node.dimensions, builder, loc), distribution)
                    .getResult();
            }
    };

    class RngBitGenerator : public Operation::MultiFor<Mln::Sdk::Functional::RngBitGenerator> {
        public:
            std::vector<mlir::Value> emit_typed(const Mln::Sdk::Functional::RngBitGenerator &node,
                                                Operations &ctx, mlir::OpBuilder &builder,
                                                mlir::Location loc) const override {
                mlir::Value state = ctx.resolve(node.initial_state, builder, loc);
                auto output_type = mlir::RankedTensorType::get(
                    mlir_shape(node.output_shape),
                    dtype_to_mlir_type(node.output_dtype, *builder.getContext()));
                mlir::stablehlo::RngAlgorithm algorithm = mlir::stablehlo::RngAlgorithm::DEFAULT;
                switch (node.algorithm) {
                case Mln::Sdk::Functional::RngAlgorithm::Default:
                    algorithm = mlir::stablehlo::RngAlgorithm::DEFAULT;
                    break;
                case Mln::Sdk::Functional::RngAlgorithm::ThreeFry:
                    algorithm = mlir::stablehlo::RngAlgorithm::THREE_FRY;
                    break;
                case Mln::Sdk::Functional::RngAlgorithm::Philox:
                    algorithm = mlir::stablehlo::RngAlgorithm::PHILOX;
                    break;
                }
                auto operation = mlir::stablehlo::RngBitGeneratorOp::create(
                    builder, loc, state.getType(), output_type, algorithm, state);
                return {operation.getOutputState(), operation.getOutput()};
            }
    };

    class Scatter : public Operation::For<Mln::Sdk::Functional::Scatter> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Scatter &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                mlir::Value input = ctx.resolve(node.input, builder, loc);
                auto dimensions = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
                    builder.getContext(), node.update_window_dimensions,
                    node.inserted_window_dimensions, node.input_batching_dimensions,
                    node.indices_batching_dimensions, node.scatter_to_operand_dimensions,
                    node.index_vector_dimension);
                auto operation = mlir::stablehlo::ScatterOp::create(
                    builder, loc, mlir::ValueRange{input}, ctx.resolve(node.indices, builder, loc),
                    mlir::ValueRange{ctx.resolve(node.updates, builder, loc)}, dimensions,
                    node.indices_are_sorted, node.unique_indices);

                auto element = mlir::cast<mlir::ShapedType>(input.getType()).getElementType();
                auto scalar = mlir::RankedTensorType::get({}, element);
                mlir::OpBuilder::InsertionGuard guard(builder);
                mlir::Block *block = builder.createBlock(&operation.getUpdateComputation(), {},
                                                         {scalar, scalar}, {loc, loc});
                builder.setInsertionPointToStart(block);
                mlir::Value lhs = block->getArgument(0);
                mlir::Value rhs = block->getArgument(1);
                mlir::Value result;
                switch (node.update) {
                case Mln::Sdk::Functional::ScatterUpdate::Replace:
                    result = rhs;
                    break;
                case Mln::Sdk::Functional::ScatterUpdate::Add:
                    result = mlir::stablehlo::AddOp::create(builder, loc, lhs, rhs);
                    break;
                case Mln::Sdk::Functional::ScatterUpdate::Multiply:
                    result = mlir::stablehlo::MulOp::create(builder, loc, lhs, rhs);
                    break;
                case Mln::Sdk::Functional::ScatterUpdate::Maximum:
                    result = mlir::stablehlo::MaxOp::create(builder, loc, lhs, rhs);
                    break;
                case Mln::Sdk::Functional::ScatterUpdate::Minimum:
                    result = mlir::stablehlo::MinOp::create(builder, loc, lhs, rhs);
                    break;
                }
                mlir::stablehlo::ReturnOp::create(builder, loc, result);
                return operation.getResult(0);
            }
    };
}
