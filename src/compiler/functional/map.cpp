#include "syn/functional/map.hpp"
#include "mln/error.hpp"
#include "syn/errors/graph.hpp"

#include "mln/sdk/functional.hpp"
#include "mln/sdk/functional/bitcastconvert.hpp"
#include "mln/sdk/functional/const.hpp"
#include "mln/sdk/functional/convert.hpp"
#include "mln/sdk/functional/reshape.hpp"
#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineMap.h>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <unordered_map>

namespace Syn::Compiler::Functional {
    const Mln::Sdk::Functional::Map *row_addressable_map(Mln::Sdk::Tensor decoded) {
        while (decoded.def && decoded.def->kind == Mln::Sdk::Functional::Kind::Reshape) {
            const auto &reshape = static_cast<const Mln::Sdk::Functional::Reshape &>(*decoded.def);
            if (decoded.shape.empty() || reshape.x.shape.empty() ||
                decoded.shape.front() != reshape.x.shape.front())
                return nullptr;
            decoded = reshape.x;
        }
        if (!decoded.def || decoded.def->kind != Mln::Sdk::Functional::Kind::Map)
            return nullptr;
        return &static_cast<const Mln::Sdk::Functional::Map &>(*decoded.def);
    }

    namespace {
        class ScalarBody {
            public:
                ScalarBody(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Type computation_float_type = {})
                    : builder_(builder), location_(location),
                      computation_float_type_(computation_float_type) {}

                void seed(const Mln::Sdk::Operation *operation, mlir::Value value) {
                    if (computation_float_type_ && mlir::isa<mlir::FloatType>(value.getType()) &&
                        value.getType() != computation_float_type_)
                        value = convert(value, computation_float_type_);
                    values_.emplace(operation, value);
                }

                mlir::Value emit(const Mln::Sdk::Tensor &tensor) {
                    if (const auto found = values_.find(tensor.def.get()); found != values_.end())
                        return found->second;

                    mlir::Value result;
                    switch (tensor.def->kind) {
                    case Mln::Sdk::Functional::Kind::Const: {
                        const auto &constant =
                            static_cast<const Mln::Sdk::Functional::Const &>(*tensor.def);
                        if (!constant.weight.value || !constant.dtype)
                            Mln::raise(Errors::Graph::Map::Scalar::Constant::Incomplete);
                        mlir::Type type =
                            dtype_to_mlir_type(*constant.dtype, *builder_.getContext());
                        if (computation_float_type_ && mlir::isa<mlir::FloatType>(type))
                            type = computation_float_type_;
                        result =
                            mlir::isa<mlir::IntegerType>(type)
                                ? mlir::arith::ConstantOp::create(
                                      builder_, location_,
                                      mlir::IntegerAttr::get(
                                          type, static_cast<std::int64_t>(*constant.weight.value)))
                                : mlir::arith::ConstantOp::create(
                                      builder_, location_,
                                      mlir::FloatAttr::get(type, *constant.weight.value));
                        break;
                    }
                    case Mln::Sdk::Functional::Kind::Convert: {
                        const auto &node =
                            static_cast<const Mln::Sdk::Functional::Convert &>(*tensor.def);
                        mlir::Type target = dtype_to_mlir_type(node.target, *builder_.getContext());
                        if (computation_float_type_ && mlir::isa<mlir::FloatType>(target))
                            target = computation_float_type_;
                        result = convert(emit(node.x), target);
                        break;
                    }
                    case Mln::Sdk::Functional::Kind::Add:
                    case Mln::Sdk::Functional::Kind::Subtract:
                    case Mln::Sdk::Functional::Kind::Multiply:
                    case Mln::Sdk::Functional::Kind::Divide:
                    case Mln::Sdk::Functional::Kind::BitwiseAnd:
                    case Mln::Sdk::Functional::Kind::ShiftLeftLogical:
                    case Mln::Sdk::Functional::Kind::ShiftRightLogical:
                        result = emit_binary(*tensor.def);
                        break;
                    default:
                        Mln::raise(Errors::Graph::Map::Operation::Unsupported);
                    }
                    values_.emplace(tensor.def.get(), result);
                    return result;
                }

            private:
                mlir::Value convert(mlir::Value value, mlir::Type target) {
                    if (const auto integer = mlir::dyn_cast<mlir::IntegerType>(target))
                        target = mlir::IntegerType::get(builder_.getContext(), integer.getWidth());
                    const mlir::Type source = value.getType();
                    if (source == target)
                        return value;
                    if (mlir::isa<mlir::IntegerType>(source) &&
                        mlir::isa<mlir::IntegerType>(target)) {
                        const unsigned source_width =
                            mlir::cast<mlir::IntegerType>(source).getWidth();
                        const unsigned target_width =
                            mlir::cast<mlir::IntegerType>(target).getWidth();
                        if (source_width < target_width)
                            return mlir::arith::ExtUIOp::create(builder_, location_, target, value);
                        if (source_width > target_width)
                            return mlir::arith::TruncIOp::create(builder_, location_, target,
                                                                 value);
                    }
                    if (mlir::isa<mlir::IntegerType>(source) && mlir::isa<mlir::FloatType>(target))
                        return mlir::arith::UIToFPOp::create(builder_, location_, target, value);
                    if (mlir::isa<mlir::FloatType>(source) && mlir::isa<mlir::FloatType>(target)) {
                        const unsigned source_width =
                            mlir::cast<mlir::FloatType>(source).getWidth();
                        const unsigned target_width =
                            mlir::cast<mlir::FloatType>(target).getWidth();
                        if (source_width < target_width)
                            return mlir::arith::ExtFOp::create(builder_, location_, target, value);
                        return mlir::arith::TruncFOp::create(builder_, location_, target, value);
                    }
                    Mln::raise(Errors::Graph::Map::Scalar::Conversion::Unsupported);
                }

                mlir::Value emit_binary(const Mln::Sdk::Operation &operation) {
                    mlir::Value left;
                    mlir::Value right;
                    const auto resolve = [&](const auto &node) {
                        left = emit(node.lhs);
                        right = emit(node.rhs);
                    };
                    switch (operation.kind) {
                    case Mln::Sdk::Functional::Kind::Add:
                        resolve(static_cast<const Mln::Sdk::Functional::Add &>(operation));
                        break;
                    case Mln::Sdk::Functional::Kind::Subtract:
                        resolve(static_cast<const Mln::Sdk::Functional::Subtract &>(operation));
                        break;
                    case Mln::Sdk::Functional::Kind::Multiply:
                        resolve(static_cast<const Mln::Sdk::Functional::Multiply &>(operation));
                        break;
                    case Mln::Sdk::Functional::Kind::Divide:
                        resolve(static_cast<const Mln::Sdk::Functional::Divide &>(operation));
                        break;
                    case Mln::Sdk::Functional::Kind::BitwiseAnd:
                        resolve(static_cast<const Mln::Sdk::Functional::BitwiseAnd &>(operation));
                        break;
                    case Mln::Sdk::Functional::Kind::ShiftLeftLogical:
                        resolve(
                            static_cast<const Mln::Sdk::Functional::ShiftLeftLogical &>(operation));
                        break;
                    case Mln::Sdk::Functional::Kind::ShiftRightLogical:
                        resolve(static_cast<const Mln::Sdk::Functional::ShiftRightLogical &>(
                            operation));
                        break;
                    default:
                        Mln::raise(Errors::Graph::Map::Operation::Unsupported);
                    }

                    const bool floating = mlir::isa<mlir::FloatType>(left.getType());
                    switch (operation.kind) {
                    case Mln::Sdk::Functional::Kind::Add:
                        return floating
                                   ? mlir::arith::AddFOp::create(builder_, location_, left, right)
                                         .getResult()
                                   : mlir::arith::AddIOp::create(builder_, location_, left, right)
                                         .getResult();
                    case Mln::Sdk::Functional::Kind::Subtract:
                        return floating
                                   ? mlir::arith::SubFOp::create(builder_, location_, left, right)
                                         .getResult()
                                   : mlir::arith::SubIOp::create(builder_, location_, left, right)
                                         .getResult();
                    case Mln::Sdk::Functional::Kind::Multiply:
                        return floating
                                   ? mlir::arith::MulFOp::create(builder_, location_, left, right)
                                         .getResult()
                                   : mlir::arith::MulIOp::create(builder_, location_, left, right)
                                         .getResult();
                    case Mln::Sdk::Functional::Kind::Divide:
                        if (!floating)
                            Mln::raise(Errors::Graph::Map::Integer::Division::Ambiguous);
                        return mlir::arith::DivFOp::create(builder_, location_, left, right);
                    case Mln::Sdk::Functional::Kind::BitwiseAnd:
                        return mlir::arith::AndIOp::create(builder_, location_, left, right);
                    case Mln::Sdk::Functional::Kind::ShiftLeftLogical:
                        return mlir::arith::ShLIOp::create(builder_, location_, left, right);
                    case Mln::Sdk::Functional::Kind::ShiftRightLogical:
                        return mlir::arith::ShRUIOp::create(builder_, location_, left, right);
                    default:
                        Mln::raise(Errors::Graph::Map::Operation::Unsupported);
                    }
                }

                mlir::OpBuilder &builder_;
                mlir::Location location_;
                mlir::Type computation_float_type_;
                std::unordered_map<const Mln::Sdk::Operation *, mlir::Value> values_;
        };

        mlir::Value gather(mlir::OpBuilder &builder, mlir::Location location, mlir::Value source,
                           mlir::Value indices, std::int64_t axis) {
            const auto source_type = mlir::cast<mlir::RankedTensorType>(source.getType());
            const auto indices_type = mlir::cast<mlir::RankedTensorType>(indices.getType());
            if (indices_type.getRank() != 1)
                Mln::raise(Errors::Graph::Map::Gather::Indices::Rank::Invalid);
            std::vector<std::int64_t> slice_sizes(source_type.getShape().begin(),
                                                  source_type.getShape().end());
            slice_sizes[axis] = 1;
            std::vector<std::int64_t> offset_dimensions;
            for (std::int64_t dimension = 0; dimension < axis; ++dimension)
                offset_dimensions.push_back(dimension);
            for (std::int64_t dimension = axis + 1; dimension < source_type.getRank(); ++dimension)
                offset_dimensions.push_back(dimension);
            const auto dimensions = mlir::stablehlo::GatherDimensionNumbersAttr::get(
                builder.getContext(), offset_dimensions, {axis}, {}, {}, {axis}, 1);
            std::vector<std::int64_t> result_shape(source_type.getShape().begin(),
                                                   source_type.getShape().end());
            result_shape[axis] = indices_type.getDimSize(0);
            return mlir::stablehlo::GatherOp::create(
                       builder, location,
                       mlir::RankedTensorType::get(result_shape, source_type.getElementType()),
                       source, indices, dimensions, slice_sizes, false)
                .getResult();
        }

        mlir::Value emit_map(const Mln::Sdk::Functional::Map &map, mlir::Value indices,
                             std::int64_t gathered_axis, Operations &context,
                             mlir::OpBuilder &builder, mlir::Location location) {
            std::vector<std::int64_t> output_shape = map.output_shape;
            if (indices) {
                const auto indices_type = mlir::cast<mlir::RankedTensorType>(indices.getType());
                if (gathered_axis < 0 ||
                    gathered_axis >= static_cast<std::int64_t>(output_shape.size()) ||
                    indices_type.getRank() != 1)
                    Mln::raise(Errors::Graph::Map::Gather::Unsupported);
                output_shape[gathered_axis] = indices_type.getDimSize(0);
            }

            std::vector<mlir::Value> inputs;
            std::vector<mlir::AffineMap> indexing_maps;
            inputs.reserve(map.operands.size());
            indexing_maps.reserve(map.operands.size() + 1);
            const unsigned rank = static_cast<unsigned>(output_shape.size());
            for (const auto &operand : map.operands) {
                mlir::Value value = context.resolve(operand.value, builder, location);
                if (indices) {
                    const auto found = std::find(operand.dimensions.begin(),
                                                 operand.dimensions.end(), gathered_axis);
                    if (found != operand.dimensions.end())
                        value = gather(builder, location, value, indices,
                                       std::distance(operand.dimensions.begin(), found));
                }
                inputs.push_back(value);
                std::vector<mlir::AffineExpr> results;
                results.reserve(operand.dimensions.size());
                for (const std::int64_t dimension : operand.dimensions) {
                    if (dimension < 0 || dimension >= static_cast<std::int64_t>(rank))
                        Mln::raise(Errors::Graph::Map::Axis::Invalid);
                    results.push_back(builder.getAffineDimExpr(static_cast<unsigned>(dimension)));
                }
                indexing_maps.push_back(
                    mlir::AffineMap::get(rank, 0, results, builder.getContext()));
            }
            indexing_maps.push_back(
                mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext()));

            const mlir::Type element_type =
                dtype_to_mlir_type(map.output_dtype, *builder.getContext());
            const mlir::Value empty =
                mlir::tensor::EmptyOp::create(builder, location, output_shape, element_type)
                    .getResult();
            const std::vector<mlir::utils::IteratorType> iterators(
                rank, mlir::utils::IteratorType::parallel);
            auto generic = mlir::linalg::GenericOp::create(
                builder, location, empty.getType(), inputs, empty, indexing_maps, iterators,
                [&](mlir::OpBuilder &nested, mlir::Location nested_location,
                    mlir::ValueRange arguments) {
                    ScalarBody body(nested, nested_location);
                    for (std::size_t index = 0; index < map.body_arguments.size(); ++index)
                        body.seed(map.body_arguments[index].def.get(), arguments[index]);
                    mlir::linalg::YieldOp::create(nested, nested_location, body.emit(map.result));
                });
            return generic.getResult(0);
        }
    }

    mlir::Value emit_map_scalar(const Mln::Sdk::Functional::Map &map, mlir::ValueRange arguments,
                                mlir::OpBuilder &builder, mlir::Location location,
                                mlir::Type computation_float_type) {
        if (arguments.size() != map.body_arguments.size())
            Mln::raise(Errors::Graph::Map::Operation::Unsupported);
        ScalarBody body(builder, location, computation_float_type);
        for (std::size_t index = 0; index < map.body_arguments.size(); ++index)
            body.seed(map.body_arguments[index].def.get(), arguments[index]);
        return body.emit(map.result);
    }

    mlir::Value Map::emit_typed(const Mln::Sdk::Functional::Map &map, Operations &ctx,
                                mlir::OpBuilder &builder, mlir::Location loc) const {
        return emit_map(map, {}, -1, ctx, builder, loc);
    }

    mlir::Value emit_gathered_map(const Mln::Sdk::Functional::Map &map, mlir::Value indices,
                                  int64_t axis, Operations &ctx, mlir::OpBuilder &builder,
                                  mlir::Location loc) {
        return emit_map(map, indices, axis, ctx, builder, loc);
    }
}
