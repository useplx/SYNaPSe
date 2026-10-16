#include "syn/functional/matmul.hpp"
#include "mln/sdk/functional/const.hpp"
#include "mln/sdk/functional/map.hpp"
#include "mln/sdk/functional/matmul.hpp"
#include "mln/sdk/functional/reshape.hpp"
#include "mln/sdk/functional/slice.hpp"
#include "mln/sdk/functional/transpose.hpp"
#include "syn/functional/map.hpp"
#include "syn/operations.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace Syn::Compiler::Functional {
    namespace {
        const Mln::Sdk::Functional::Map *mapped_weight(Mln::Sdk::Tensor value) {
            while (value.def && value.def->kind == Mln::Sdk::Functional::Kind::Reshape)
                value = static_cast<const Mln::Sdk::Functional::Reshape &>(*value.def).x;
            if (!value.def || value.def->kind != Mln::Sdk::Functional::Kind::Map)
                return nullptr;
            return &static_cast<const Mln::Sdk::Functional::Map &>(*value.def);
        }

        mlir::Value slice_rows(mlir::Value value, int64_t axis, int64_t begin, int64_t end,
                               mlir::OpBuilder &builder, mlir::Location loc) {
            const auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
            std::vector<int64_t> starts(static_cast<std::size_t>(type.getRank()), 0);
            std::vector<int64_t> limits(type.getShape().begin(), type.getShape().end());
            std::vector<int64_t> strides(static_cast<std::size_t>(type.getRank()), 1);
            starts[static_cast<std::size_t>(axis)] = begin;
            limits[static_cast<std::size_t>(axis)] = end;
            return mlir::stablehlo::SliceOp::create(
                       builder, loc, value,
                       mlir::DenseI64ArrayAttr::get(builder.getContext(), starts),
                       mlir::DenseI64ArrayAttr::get(builder.getContext(), limits),
                       mlir::DenseI64ArrayAttr::get(builder.getContext(), strides))
                .getResult();
        }

        mlir::Value slice_rows(mlir::Value value, int64_t axis, mlir::Value begin, int64_t size,
                               mlir::OpBuilder &builder, mlir::Location loc) {
            const auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
            const auto scalar_type = mlir::RankedTensorType::get({}, builder.getI32Type());
            mlir::Value zero =
                mlir::stablehlo::ConstantOp::create(
                    builder, loc,
                    mlir::DenseElementsAttr::get(scalar_type, builder.getI32IntegerAttr(0)))
                    .getResult();
            std::vector<mlir::Value> starts(static_cast<std::size_t>(type.getRank()), zero);
            std::vector<int64_t> sizes(type.getShape().begin(), type.getShape().end());
            starts[static_cast<std::size_t>(axis)] = begin;
            sizes[static_cast<std::size_t>(axis)] = size;
            return mlir::stablehlo::DynamicSliceOp::create(builder, loc, value, starts, sizes)
                .getResult();
        }

        mlir::Value reduce_pairs(mlir::Value value, unsigned dimensions, mlir::OpBuilder &builder,
                                 mlir::Location loc) {
            auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
            std::vector<int64_t> shape(type.getShape().begin(), type.getShape().end());
            for (unsigned dimension = 0; dimension < dimensions; ++dimension) {
                while (shape.back() > 1) {
                    const int64_t extent = shape.back();
                    if (extent % 2 != 0)
                        return {};
                    std::vector<int64_t> paired_shape(shape.begin(), shape.end() - 1);
                    paired_shape.push_back(extent / 2);
                    paired_shape.push_back(2);
                    value =
                        mlir::stablehlo::ReshapeOp::create(
                            builder, loc,
                            mlir::RankedTensorType::get(paired_shape, type.getElementType()), value)
                            .getResult();

                    std::vector<int64_t> starts(paired_shape.size(), 0);
                    std::vector<int64_t> limits(paired_shape.begin(), paired_shape.end());
                    std::vector<int64_t> strides(paired_shape.size(), 1);
                    limits.back() = 1;
                    mlir::Value even =
                        mlir::stablehlo::SliceOp::create(
                            builder, loc, value,
                            mlir::DenseI64ArrayAttr::get(builder.getContext(), starts),
                            mlir::DenseI64ArrayAttr::get(builder.getContext(), limits),
                            mlir::DenseI64ArrayAttr::get(builder.getContext(), strides))
                            .getResult();
                    starts.back() = 1;
                    limits.back() = 2;
                    mlir::Value odd =
                        mlir::stablehlo::SliceOp::create(
                            builder, loc, value,
                            mlir::DenseI64ArrayAttr::get(builder.getContext(), starts),
                            mlir::DenseI64ArrayAttr::get(builder.getContext(), limits),
                            mlir::DenseI64ArrayAttr::get(builder.getContext(), strides))
                            .getResult();
                    paired_shape.pop_back();
                    const auto next_type =
                        mlir::RankedTensorType::get(paired_shape, type.getElementType());
                    even = mlir::stablehlo::ReshapeOp::create(builder, loc, next_type, even)
                               .getResult();
                    odd = mlir::stablehlo::ReshapeOp::create(builder, loc, next_type, odd)
                              .getResult();
                    value = mlir::stablehlo::AddOp::create(builder, loc, even, odd).getResult();
                    shape = paired_shape;
                    type = next_type;
                }
                shape.pop_back();
                type = mlir::RankedTensorType::get(shape, type.getElementType());
                value = mlir::stablehlo::ReshapeOp::create(builder, loc, type, value).getResult();
            }
            return value;
        }

        [[maybe_unused]] mlir::Value
        emit_pairwise_mapped_dot(mlir::Value lhs, const Mln::Sdk::Functional::Map &map,
                                 int64_t begin, int64_t end, Operations &ctx,
                                 mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value dynamic_begin) {
            auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
            const bool vector_lhs = lhs_type.getRank() == 1;
            const int64_t lhs_rows = vector_lhs ? 1 : lhs_type.getDimSize(0);
            const int64_t rows = end - begin;
            const unsigned reductions = static_cast<unsigned>(map.output_shape.size() - 1);
            const unsigned loops = 3 + reductions;
            std::vector<int64_t> lhs_shape{1, lhs_rows};
            lhs_shape.insert(lhs_shape.end(), map.output_shape.begin() + 1, map.output_shape.end());
            lhs = mlir::stablehlo::ReshapeOp::create(
                      builder, loc,
                      mlir::RankedTensorType::get(lhs_shape, lhs_type.getElementType()), lhs)
                      .getResult();

            const auto d = [&](unsigned axis) { return builder.getAffineDimExpr(axis); };
            std::vector<mlir::Value> inputs{lhs};
            std::vector<mlir::AffineExpr> lhs_map{d(0), d(1)};
            for (unsigned axis = 0; axis < reductions; ++axis)
                lhs_map.push_back(d(3 + axis));
            std::vector<mlir::AffineMap> maps{
                mlir::AffineMap::get(loops, 0, lhs_map, builder.getContext())};
            for (const auto &operand : map.operands) {
                mlir::Value value = ctx.resolve(operand.value, builder, loc);
                std::vector<mlir::AffineExpr> operand_map;
                for (std::size_t axis = 0; axis < operand.dimensions.size(); ++axis) {
                    const int64_t dimension = operand.dimensions[axis];
                    if (dimension == 0) {
                        if (dynamic_begin)
                            value = slice_rows(value, static_cast<int64_t>(axis), dynamic_begin,
                                               rows, builder, loc);
                        else if (begin != 0 || end != map.output_shape.front())
                            value = slice_rows(value, static_cast<int64_t>(axis), begin, end,
                                               builder, loc);
                        operand_map.push_back(d(2));
                    } else {
                        operand_map.push_back(d(static_cast<unsigned>(2 + dimension)));
                    }
                }
                inputs.push_back(value);
                maps.push_back(mlir::AffineMap::get(loops, 0, operand_map, builder.getContext()));
            }
            std::vector<mlir::AffineExpr> output_map{d(0), d(1), d(2)};
            for (unsigned axis = 0; axis < reductions; ++axis)
                output_map.push_back(d(3 + axis));
            maps.push_back(mlir::AffineMap::get(loops, 0, output_map, builder.getContext()));

            const auto accumulation_type = builder.getF32Type();
            std::vector<int64_t> products_shape{1, lhs_rows, rows};
            products_shape.insert(products_shape.end(), map.output_shape.begin() + 1,
                                  map.output_shape.end());
            const auto products_type =
                mlir::RankedTensorType::get(products_shape, accumulation_type);
            mlir::Value empty = mlir::tensor::EmptyOp::create(
                                    builder, loc, products_type.getShape(), accumulation_type)
                                    .getResult();
            mlir::Value products =
                mlir::linalg::GenericOp::create(
                    builder, loc, products_type, inputs, empty, maps,
                    std::vector<mlir::utils::IteratorType>(loops,
                                                           mlir::utils::IteratorType::parallel),
                    [&](mlir::OpBuilder &nested, mlir::Location nested_loc,
                        mlir::ValueRange arguments) {
                        mlir::Value decoded = emit_map_scalar(
                            map, arguments.slice(1, map.operands.size()), nested, nested_loc);
                        mlir::Value activation = arguments.front();
                        if (activation.getType() != accumulation_type)
                            activation = mlir::arith::ExtFOp::create(nested, nested_loc,
                                                                     accumulation_type, activation);
                        if (decoded.getType() != accumulation_type)
                            decoded = mlir::arith::ExtFOp::create(nested, nested_loc,
                                                                  accumulation_type, decoded);
                        mlir::Value product =
                            mlir::arith::MulFOp::create(nested, nested_loc, activation, decoded);
                        mlir::linalg::YieldOp::create(nested, nested_loc, product);
                    })
                    .getResult(0);
            products = reduce_pairs(products, reductions, builder, loc);
            if (!products)
                return {};
            const std::vector<int64_t> result_shape =
                vector_lhs ? std::vector<int64_t>{rows} : std::vector<int64_t>{lhs_rows, rows};
            mlir::Value result =
                mlir::stablehlo::ReshapeOp::create(
                    builder, loc, mlir::RankedTensorType::get(result_shape, accumulation_type),
                    products)
                    .getResult();
            if (lhs_type.getElementType() == accumulation_type)
                return result;
            return mlir::stablehlo::ConvertOp::create(
                       builder, loc,
                       mlir::RankedTensorType::get(result_shape, lhs_type.getElementType()), result)
                .getResult();
        }

        mlir::Value emit_mapped_dot(mlir::Value lhs, const Mln::Sdk::Functional::Map &map,
                                    int64_t begin, int64_t end, bool linearize_reduction,
                                    int64_t reduction_tile, Operations &ctx,
                                    mlir::OpBuilder &builder, mlir::Location loc,
                                    mlir::Value dynamic_begin = {}) {
            auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
            if (!lhs_type || (lhs_type.getRank() != 1 && lhs_type.getRank() != 2) ||
                !lhs_type.hasStaticShape() || map.output_shape.size() < 3 ||
                map.output_shape.front() <= 0)
                return {};

            const bool vector_lhs = lhs_type.getRank() == 1;
            const int64_t lhs_rows = vector_lhs ? 1 : lhs_type.getDimSize(0);
            int64_t contracted = 1;
            for (std::size_t axis = 1; axis < map.output_shape.size(); ++axis)
                contracted *= map.output_shape[axis];
            if (lhs_type.getDimSize(lhs_type.getRank() - 1) != contracted || begin < 0 ||
                end <= begin || end > map.output_shape.front())
                return {};

            const bool segmented = linearize_reduction && reduction_tile > 0 &&
                                   contracted > reduction_tile && contracted % reduction_tile == 0;
            const int64_t chunks = segmented ? contracted / reduction_tile : 1;
            int64_t split_axis = -1;
            int64_t split_stride = 1;
            int64_t split_count = 1;
            if (segmented) {
                for (int64_t axis = static_cast<int64_t>(map.output_shape.size()) - 1; axis >= 1;
                     --axis) {
                    if (reduction_tile % split_stride == 0 &&
                        reduction_tile / split_stride <= map.output_shape[axis]) {
                        split_axis = axis;
                        split_count = reduction_tile / split_stride;
                        break;
                    }
                    split_stride *= map.output_shape[axis];
                }
                if (split_axis < 1 || map.output_shape[split_axis] % split_count != 0 ||
                    map.output_shape[split_axis] / split_count != chunks)
                    return {};
            }
            std::vector<int64_t> lhs_shape{1, lhs_rows};
            if (segmented) {
                lhs_shape.push_back(chunks);
                lhs_shape.push_back(reduction_tile);
            } else if (linearize_reduction) {
                lhs_shape.push_back(contracted);
            } else {
                lhs_shape.insert(lhs_shape.end(), map.output_shape.begin() + 1,
                                 map.output_shape.end());
            }
            lhs = mlir::stablehlo::ReshapeOp::create(
                builder, loc, mlir::RankedTensorType::get(lhs_shape, lhs_type.getElementType()),
                lhs);

            std::vector<mlir::Value> inputs{lhs};
            std::vector<mlir::AffineMap> maps;
            const unsigned reductions =
                linearize_reduction ? 1 : static_cast<unsigned>(map.output_shape.size() - 1);
            const unsigned loop_count = 3 + reductions + (segmented ? 1 : 0);
            const auto d = [&](unsigned axis) { return builder.getAffineDimExpr(axis); };
            std::vector<mlir::AffineExpr> lhs_map =
                segmented ? std::vector<mlir::AffineExpr>{d(1), d(2), d(0), d(4)}
                          : std::vector<mlir::AffineExpr>{d(0), d(1)};
            if (!segmented) {
                for (unsigned axis = 0; axis < reductions; ++axis)
                    lhs_map.push_back(d(3 + axis));
            }
            maps.push_back(mlir::AffineMap::get(loop_count, 0, lhs_map, builder.getContext()));

            const auto flattened_axis = [&](int64_t dimension) -> mlir::AffineExpr {
                mlir::AffineExpr index = segmented ? d(4) : d(3);
                int64_t stride = 1;
                for (std::size_t axis = static_cast<std::size_t>(dimension + 1);
                     axis < map.output_shape.size(); ++axis)
                    stride *= map.output_shape[axis];
                if (stride != 1)
                    index = index.floorDiv(static_cast<uint64_t>(stride));
                const int64_t extent = segmented && dimension == split_axis
                                           ? split_count
                                           : map.output_shape[dimension];
                return index % static_cast<uint64_t>(extent);
            };

            for (const auto &operand : map.operands) {
                mlir::Value value = ctx.resolve(operand.value, builder, loc);
                std::vector<mlir::AffineExpr> operand_map;
                operand_map.reserve(operand.dimensions.size() + (segmented ? 1 : 0));
                for (std::size_t axis = 0; axis < operand.dimensions.size(); ++axis) {
                    const int64_t dimension = operand.dimensions[axis];
                    if (dimension < 0 || dimension >= static_cast<int64_t>(map.output_shape.size()))
                        return {};
                    if (dimension == 0) {
                        if (dynamic_begin)
                            value = slice_rows(value, static_cast<int64_t>(axis), dynamic_begin,
                                               end - begin, builder, loc);
                        else if (begin != 0 || end != map.output_shape.front())
                            value = slice_rows(value, static_cast<int64_t>(axis), begin, end,
                                               builder, loc);
                    }
                    if (segmented && dimension == split_axis) {
                        auto value_type = mlir::cast<mlir::RankedTensorType>(value.getType());
                        std::vector<int64_t> expanded_shape(value_type.getShape().begin(),
                                                            value_type.getShape().end());
                        expanded_shape[axis] = chunks;
                        expanded_shape.insert(expanded_shape.begin() + axis + 1, split_count);
                        value = mlir::stablehlo::ReshapeOp::create(
                            builder, loc,
                            mlir::RankedTensorType::get(expanded_shape,
                                                        value_type.getElementType()),
                            value);
                        operand_map.push_back(d(0));
                        operand_map.push_back(flattened_axis(dimension));
                        continue;
                    }
                    operand_map.push_back(dimension == 0
                                              ? d(segmented ? 3 : 2)
                                              : (linearize_reduction
                                                     ? flattened_axis(dimension)
                                                     : d(static_cast<unsigned>(2 + dimension))));
                }
                inputs.push_back(value);
                maps.push_back(
                    mlir::AffineMap::get(loop_count, 0, operand_map, builder.getContext()));
            }
            maps.push_back(mlir::AffineMap::get(
                loop_count, 0,
                segmented ? std::vector<mlir::AffineExpr>{d(0), d(1), d(2), d(3)}
                          : std::vector<mlir::AffineExpr>{d(0), d(1), d(2)},
                builder.getContext()));

            const int64_t rows = end - begin;
            const mlir::Type accumulation_type = builder.getF32Type();
            auto grouped_type = mlir::RankedTensorType::get(
                segmented ? std::vector<int64_t>{chunks, 1, lhs_rows, rows}
                          : std::vector<int64_t>{1, lhs_rows, rows},
                accumulation_type);
            mlir::Value empty = mlir::tensor::EmptyOp::create(builder, loc, grouped_type.getShape(),
                                                              grouped_type.getElementType())
                                    .getResult();
            mlir::Value zero = mlir::arith::ConstantOp::create(
                builder, loc, mlir::FloatAttr::get(grouped_type.getElementType(), 0.0));
            mlir::Value initialized =
                mlir::linalg::FillOp::create(builder, loc, zero, empty).getResult(0);
            std::vector<mlir::utils::IteratorType> iterators{
                mlir::utils::IteratorType::parallel,
                mlir::utils::IteratorType::parallel,
                mlir::utils::IteratorType::parallel,
            };
            if (segmented)
                iterators.push_back(mlir::utils::IteratorType::parallel);
            iterators.insert(iterators.end(), reductions, mlir::utils::IteratorType::reduction);
            mlir::Value grouped =
                mlir::linalg::GenericOp::create(
                    builder, loc, grouped_type, inputs, initialized, maps, iterators,
                    [&](mlir::OpBuilder &nested, mlir::Location nested_loc,
                        mlir::ValueRange arguments) {
                        mlir::Value decoded = emit_map_scalar(
                            map, arguments.slice(1, map.operands.size()), nested, nested_loc);
                        mlir::Value activation = arguments.front();
                        if (activation.getType() != accumulation_type)
                            activation = mlir::arith::ExtFOp::create(nested, nested_loc,
                                                                     accumulation_type, activation);
                        if (decoded.getType() != accumulation_type)
                            decoded = mlir::arith::ExtFOp::create(nested, nested_loc,
                                                                  accumulation_type, decoded);
                        mlir::Value product =
                            mlir::arith::MulFOp::create(nested, nested_loc, activation, decoded);
                        mlir::Value sum = mlir::arith::AddFOp::create(nested, nested_loc, product,
                                                                      arguments.back());
                        mlir::linalg::YieldOp::create(nested, nested_loc, sum);
                    })
                    .getResult(0);
            if (segmented) {
                const auto reduced_type =
                    mlir::RankedTensorType::get({1, lhs_rows, rows}, accumulation_type);
                const auto scalar_type = mlir::RankedTensorType::get({}, accumulation_type);
                mlir::Value initial =
                    mlir::stablehlo::ConstantOp::create(
                        builder, loc,
                        mlir::DenseElementsAttr::get(scalar_type, builder.getF32FloatAttr(0.0)))
                        .getResult();
                auto reduction = mlir::stablehlo::ReduceOp::create(
                    builder, loc, mlir::TypeRange{reduced_type}, mlir::ValueRange{grouped},
                    mlir::ValueRange{initial},
                    mlir::DenseI64ArrayAttr::get(builder.getContext(), {0}));
                mlir::OpBuilder::InsertionGuard guard(builder);
                mlir::Block *block = builder.createBlock(&reduction.getBody(), {},
                                                         {scalar_type, scalar_type}, {loc, loc});
                builder.setInsertionPointToStart(block);
                mlir::Value sum = mlir::stablehlo::AddOp::create(
                    builder, loc, block->getArgument(0), block->getArgument(1));
                mlir::stablehlo::ReturnOp::create(builder, loc, sum);
                grouped = reduction.getResult(0);
            }
            const std::vector<int64_t> result_shape =
                vector_lhs ? std::vector<int64_t>{rows} : std::vector<int64_t>{lhs_rows, rows};
            mlir::Value reshaped = mlir::stablehlo::ReshapeOp::create(
                builder, loc, mlir::RankedTensorType::get(result_shape, accumulation_type),
                grouped);
            if (lhs_type.getElementType() == accumulation_type)
                return reshaped;
            return mlir::stablehlo::ConvertOp::create(
                       builder, loc,
                       mlir::RankedTensorType::get(result_shape, lhs_type.getElementType()),
                       reshaped)
                .getResult();
        }

        mlir::Value emit_looped_mapped_dot(mlir::Value lhs, const Mln::Sdk::Functional::Map &map,
                                           int64_t tile, bool linearize_reduction,
                                           int64_t reduction_tile, Operations &ctx,
                                           mlir::OpBuilder &builder, mlir::Location loc) {
            const int64_t rows = map.output_shape.front();
            if (tile <= 0 || rows <= tile || rows > std::numeric_limits<int32_t>::max())
                return {};
            const int64_t loop_rows = rows - rows % tile;
            if (loop_rows <= 0)
                return {};

            const auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
            if (!lhs_type || !lhs_type.hasStaticShape() ||
                (lhs_type.getRank() != 1 && lhs_type.getRank() != 2) || map.output_shape.size() < 3)
                return {};
            int64_t contracted = 1;
            for (const int64_t extent : llvm::drop_begin(map.output_shape)) {
                if (extent <= 0 || contracted > std::numeric_limits<int64_t>::max() / extent)
                    return {};
                contracted *= extent;
            }
            if (lhs_type.getDimSize(lhs_type.getRank() - 1) != contracted)
                return {};
            for (const auto &operand : map.operands) {
                if (std::ranges::any_of(operand.dimensions, [&](int64_t dimension) {
                        return dimension < 0 ||
                               dimension >= static_cast<int64_t>(map.output_shape.size());
                    }))
                    return {};
            }

            for (const auto &operand : map.operands)
                ctx.resolve(operand.value, builder, loc);
            const bool vector_lhs = lhs_type.getRank() == 1;
            const int64_t lhs_rows = vector_lhs ? 1 : lhs_type.getDimSize(0);
            const std::vector<int64_t> result_shape =
                vector_lhs ? std::vector<int64_t>{loop_rows}
                           : std::vector<int64_t>{lhs_rows, loop_rows};
            const auto result_type =
                mlir::RankedTensorType::get(result_shape, lhs_type.getElementType());
            mlir::Value empty =
                mlir::tensor::EmptyOp::create(builder, loc, result_shape, lhs_type.getElementType())
                    .getResult();
            mlir::Value zero = mlir::arith::ConstantOp::create(
                builder, loc, builder.getZeroAttr(lhs_type.getElementType()));
            mlir::Value initial =
                mlir::linalg::FillOp::create(builder, loc, zero, empty).getResult(0);

            const auto scalar_type = mlir::RankedTensorType::get({}, builder.getI32Type());
            const auto scalar = [&](int32_t value) {
                return mlir::stablehlo::ConstantOp::create(
                           builder, loc,
                           mlir::DenseElementsAttr::get(scalar_type,
                                                        builder.getI32IntegerAttr(value)))
                    .getResult();
            };
            mlir::Value index = scalar(0);
            mlir::Value limit = scalar(static_cast<int32_t>(loop_rows));
            mlir::Value step = scalar(static_cast<int32_t>(tile));
            const std::array<mlir::Type, 2> types{scalar_type, result_type};
            const std::array<mlir::Value, 2> operands{index, initial};
            auto loop = mlir::stablehlo::WhileOp::create(builder, loc, types, operands);

            mlir::OpBuilder::InsertionGuard guard(builder);
            mlir::Block &condition = loop.getCond().emplaceBlock();
            condition.addArguments(types, std::array<mlir::Location, 2>{loc, loc});
            builder.setInsertionPointToStart(&condition);
            mlir::Value predicate =
                mlir::stablehlo::CompareOp::create(builder, loc, condition.getArgument(0), limit,
                                                   mlir::stablehlo::ComparisonDirection::LT,
                                                   mlir::stablehlo::ComparisonType::SIGNED)
                    .getResult();
            mlir::stablehlo::ReturnOp::create(builder, loc, predicate);

            mlir::Block &body = loop.getBody().emplaceBlock();
            body.addArguments(types, std::array<mlir::Location, 2>{loc, loc});
            builder.setInsertionPointToStart(&body);
            mlir::Value part =
                emit_mapped_dot(lhs, map, 0, tile, linearize_reduction, reduction_tile, ctx,
                                builder, loc, body.getArgument(0));
            if (!part)
                return {};
            std::vector<mlir::Value> starts;
            if (!vector_lhs)
                starts.push_back(scalar(0));
            starts.push_back(body.getArgument(0));
            mlir::Value updated = mlir::stablehlo::DynamicUpdateSliceOp::create(
                                      builder, loc, body.getArgument(1), part, starts)
                                      .getResult();
            mlir::Value next =
                mlir::stablehlo::AddOp::create(builder, loc, body.getArgument(0), step).getResult();
            mlir::stablehlo::ReturnOp::create(builder, loc, mlir::ValueRange{next, updated});
            if (loop_rows == rows)
                return loop.getResult(1);

            builder.setInsertionPointAfter(loop);
            mlir::Value tail = emit_mapped_dot(lhs, map, loop_rows, rows, linearize_reduction,
                                               reduction_tile, ctx, builder, loc);
            if (!tail)
                return {};
            const std::vector<int64_t> full_shape =
                vector_lhs ? std::vector<int64_t>{rows} : std::vector<int64_t>{lhs_rows, rows};
            const auto full_type =
                mlir::RankedTensorType::get(full_shape, lhs_type.getElementType());
            return mlir::stablehlo::ConcatenateOp::create(
                       builder, loc, full_type, mlir::ValueRange{loop.getResult(1), tail},
                       static_cast<uint64_t>(full_shape.size() - 1))
                .getResult();
        }

        mlir::Value emit_reassociated_dot(mlir::Value lhs, mlir::Value rhs,
                                          mlir::OpBuilder &builder, mlir::Location loc) {
            auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
            auto transpose = rhs.getDefiningOp<mlir::stablehlo::TransposeOp>();
            if (!lhs_type || (lhs_type.getRank() != 1 && lhs_type.getRank() != 2) || !transpose)
                return {};

            const auto permutation = transpose.getPermutation();
            if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0)
                return {};

            auto flatten = transpose.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
            if (!flatten)
                return {};
            mlir::Value grouped_rhs = flatten.getOperand();
            while (auto reshape = grouped_rhs.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
                auto candidate =
                    mlir::dyn_cast<mlir::RankedTensorType>(reshape.getOperand().getType());
                if (!candidate || !candidate.hasStaticShape() || candidate.getRank() < 3)
                    break;
                grouped_rhs = reshape.getOperand();
            }
            auto grouped_type = mlir::dyn_cast<mlir::RankedTensorType>(grouped_rhs.getType());
            auto flat_type = mlir::dyn_cast<mlir::RankedTensorType>(flatten.getType());
            if (!grouped_type || grouped_type.getRank() < 3 || !flat_type ||
                flat_type.getRank() != 2 || !grouped_type.hasStaticShape() ||
                !flat_type.hasStaticShape() || !lhs_type.hasStaticShape())
                return {};

            const int64_t rows = grouped_type.getDimSize(0);
            const bool vector_lhs = lhs_type.getRank() == 1;
            const int64_t rows_lhs = vector_lhs ? 1 : lhs_type.getDimSize(0);
            const int64_t contracted = lhs_type.getDimSize(lhs_type.getRank() - 1);
            int64_t grouped_contracted = 1;
            for (int64_t axis = 1; axis < grouped_type.getRank(); ++axis)
                grouped_contracted *= grouped_type.getDimSize(axis);
            if (grouped_contracted != contracted || flat_type.getDimSize(0) != rows ||
                flat_type.getDimSize(1) != contracted)
                return {};

            std::vector<int64_t> lhs_shape{1, rows_lhs};
            lhs_shape.insert(lhs_shape.end(), grouped_type.getShape().begin() + 1,
                             grouped_type.getShape().end());
            auto reassociated_lhs_type =
                mlir::RankedTensorType::get(lhs_shape, lhs_type.getElementType());
            lhs = mlir::stablehlo::ReshapeOp::create(builder, loc, reassociated_lhs_type, lhs);

            auto grouped_result_type =
                mlir::RankedTensorType::get({1, rows_lhs, rows}, lhs_type.getElementType());
            mlir::Value empty =
                mlir::tensor::EmptyOp::create(builder, loc, grouped_result_type.getShape(),
                                              grouped_result_type.getElementType())
                    .getResult();
            mlir::Value zero = mlir::arith::ConstantOp::create(
                builder, loc, mlir::FloatAttr::get(grouped_result_type.getElementType(), 0.0));
            mlir::Value initialized =
                mlir::linalg::FillOp::create(builder, loc, zero, empty).getResult(0);

            auto d = [&](unsigned axis) { return builder.getAffineDimExpr(axis); };
            const unsigned reductions = static_cast<unsigned>(grouped_type.getRank() - 1);
            const unsigned loop_count = 3 + reductions;
            std::vector<mlir::AffineExpr> lhs_map{d(0), d(1)};
            std::vector<mlir::AffineExpr> rhs_map{d(2)};
            for (unsigned axis = 0; axis < reductions; ++axis) {
                lhs_map.push_back(d(3 + axis));
                rhs_map.push_back(d(3 + axis));
            }
            std::vector<mlir::AffineMap> maps{
                mlir::AffineMap::get(loop_count, 0, lhs_map, builder.getContext()),
                mlir::AffineMap::get(loop_count, 0, rhs_map, builder.getContext()),
                mlir::AffineMap::get(loop_count, 0, {d(0), d(1), d(2)}, builder.getContext()),
            };
            std::vector<mlir::utils::IteratorType> iterators{
                mlir::utils::IteratorType::parallel,
                mlir::utils::IteratorType::parallel,
                mlir::utils::IteratorType::parallel,
            };
            iterators.insert(iterators.end(), reductions, mlir::utils::IteratorType::reduction);
            mlir::Value grouped_result =
                mlir::linalg::GenericOp::create(
                    builder, loc, grouped_result_type, mlir::ValueRange{lhs, grouped_rhs},
                    initialized, maps, iterators,
                    [&](mlir::OpBuilder &nested, mlir::Location nested_loc,
                        mlir::ValueRange arguments) {
                        mlir::Value product = mlir::arith::MulFOp::create(
                            nested, nested_loc, arguments[0], arguments[1]);
                        mlir::Value sum =
                            mlir::arith::AddFOp::create(nested, nested_loc, product, arguments[2]);
                        mlir::linalg::YieldOp::create(nested, nested_loc, sum);
                    })
                    .getResult(0);
            const std::vector<int64_t> result_shape =
                vector_lhs ? std::vector<int64_t>{rows} : std::vector<int64_t>{rows_lhs, rows};
            auto result_type = mlir::RankedTensorType::get(result_shape, lhs_type.getElementType());
            return mlir::stablehlo::ReshapeOp::create(builder, loc, result_type, grouped_result);
        }

        mlir::Value emit_dot(mlir::Value lhs, mlir::Value rhs, mlir::OpBuilder &builder,
                             mlir::Location loc) {
            auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
            auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
            const int64_t lhs_rank = lhs_type.getRank();
            const int64_t rhs_rank = rhs_type.getRank();

            std::vector<int64_t> result_shape(lhs_type.getShape().begin(),
                                              lhs_type.getShape().end() - 1);
            if (rhs_rank > 1)
                result_shape.push_back(rhs_type.getShape().back());

            std::vector<int64_t> lhs_batch;
            std::vector<int64_t> rhs_batch;
            if (lhs_rank > 1) {
                for (int64_t i = 0; i < lhs_rank - 2; ++i) {
                    lhs_batch.push_back(i);
                    rhs_batch.push_back(i);
                }
            }

            auto dimensions = mlir::stablehlo::DotDimensionNumbersAttr::get(
                builder.getContext(), lhs_batch, rhs_batch, {lhs_rank - 1},
                {rhs_rank > 1 ? rhs_rank - 2 : 0});
            auto result_type = mlir::RankedTensorType::get(result_shape, lhs_type.getElementType());
            return mlir::stablehlo::DotGeneralOp::create(builder, loc, result_type, lhs, rhs,
                                                         dimensions, nullptr, nullptr)
                .getResult();
        }

        mlir::Value emit_tiled_reassociated_dot(const Mln::Sdk::Functional::MatMul &matmul,
                                                int64_t tile, Operations &ctx,
                                                mlir::OpBuilder &builder, mlir::Location loc) {
            if (!matmul.rhs.def || matmul.rhs.def->kind != Mln::Sdk::Functional::Kind::Transpose)
                return {};
            const auto *transpose =
                static_cast<const Mln::Sdk::Functional::Transpose *>(matmul.rhs.def.get());
            if (transpose->permutation != std::vector<int64_t>{1, 0} ||
                transpose->x.shape.size() != 2)
                return {};

            const int64_t rows = transpose->x.shape[0];
            const int64_t contracted = transpose->x.shape[1];
            if (rows <= tile || tile <= 0)
                return {};

            mlir::Value lhs = ctx.resolve(matmul.lhs, builder, loc);
            const auto *constant =
                transpose->x.def && transpose->x.def->kind == Mln::Sdk::Functional::Kind::Const
                    ? static_cast<const Mln::Sdk::Functional::Const *>(transpose->x.def.get())
                    : nullptr;
            const Mln::Sdk::Functional::Map *map = nullptr;
            Mln::Sdk::Tensor decoded;
            if (constant && constant->quantize) {
                decoded = ctx.decode_weight(*constant);
                map = mapped_weight(decoded);
            }
            if (map && !ctx.unroll(matmul)) {
                const int64_t reduction_tile = ctx.contraction_tile(matmul, contracted);
                if (mlir::Value looped =
                        emit_looped_mapped_dot(lhs, *map, tile, ctx.linearize_reduction(matmul),
                                               reduction_tile, ctx, builder, loc))
                    return looped;
            }
            Mln::Sdk::Operations &expansions = ctx.expansion_container();
            expansions.set_policy_path(matmul.policy_path);
            std::vector<mlir::Value> parts;
            parts.reserve(static_cast<std::size_t>((rows + tile - 1) / tile));
            for (int64_t begin = 0; begin < rows; begin += tile) {
                const int64_t end = std::min(begin + tile, rows);
                if (map) {
                    const int64_t reduction_tile = ctx.contraction_tile(matmul, contracted);
                    mlir::Value part =
                        emit_mapped_dot(lhs, *map, begin, end, ctx.linearize_reduction(matmul),
                                        reduction_tile, ctx, builder, loc);
                    if (!part)
                        return {};
                    parts.push_back(part);
                    continue;
                }
                Mln::Sdk::Tensor slice = expansions.insert<Mln::Sdk::Functional::Slice>(
                    transpose->x, std::vector<int64_t>{begin, 0},
                    std::vector<int64_t>{end, contracted}, std::vector<int64_t>{1, 1});
                Mln::Sdk::Tensor rhs = expansions.insert<Mln::Sdk::Functional::Transpose>(
                    std::move(slice), std::vector<int64_t>{1, 0});
                mlir::Value emitted_rhs = ctx.resolve(rhs, builder, loc);
                mlir::Value part = emit_reassociated_dot(lhs, emitted_rhs, builder, loc);
                if (!part)
                    part = emit_dot(lhs, emitted_rhs, builder, loc);
                parts.push_back(part);
            }

            std::vector<int64_t> result_shape(matmul.signature());
            auto first_type = mlir::cast<mlir::RankedTensorType>(parts.front().getType());
            auto result_type =
                mlir::RankedTensorType::get(result_shape, first_type.getElementType());
            return mlir::stablehlo::ConcatenateOp::create(
                       builder, loc, result_type, parts,
                       static_cast<uint64_t>(result_shape.size() - 1))
                .getResult();
        }
    }

    mlir::Value MatMul::emit_typed(const Mln::Sdk::Functional::MatMul &matmul, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const {
        const std::vector<int64_t> shape = matmul.signature();
        if (!shape.empty() && shape.back() > 0) {
            const int64_t tile = ctx.tile(matmul, -1, shape.back());
            if (tile < shape.back()) {
                if (mlir::Value tiled =
                        emit_tiled_reassociated_dot(matmul, tile, ctx, builder, loc))
                    return tiled;
            }
        }
        mlir::Value lhs = ctx.resolve(matmul.lhs, builder, loc);
        if (ctx.fuse_producer(matmul) && matmul.rhs.def &&
            matmul.rhs.def->kind == Mln::Sdk::Functional::Kind::Transpose) {
            const auto &transpose =
                static_cast<const Mln::Sdk::Functional::Transpose &>(*matmul.rhs.def);
            if (transpose.permutation == std::vector<int64_t>{1, 0} && transpose.x.def &&
                transpose.x.def->kind == Mln::Sdk::Functional::Kind::Const) {
                const auto &constant =
                    static_cast<const Mln::Sdk::Functional::Const &>(*transpose.x.def);
                if (constant.quantize) {
                    Mln::Sdk::Tensor decoded = ctx.decode_weight(constant);
                    if (const auto *map = mapped_weight(decoded)) {
                        int64_t contracted = 1;
                        for (std::size_t axis = 1; axis < map->output_shape.size(); ++axis)
                            contracted *= map->output_shape[axis];
                        const int64_t reduction_tile = ctx.contraction_tile(matmul, contracted);
                        if (mlir::Value fused = emit_mapped_dot(
                                lhs, *map, 0, map->output_shape.front(),
                                ctx.linearize_reduction(matmul), reduction_tile, ctx, builder, loc))
                            return fused;
                    }
                }
            }
        }
        mlir::Value rhs = ctx.resolve(matmul.rhs, builder, loc);
        if (mlir::Value reassociated = emit_reassociated_dot(lhs, rhs, builder, loc))
            return reassociated;
        return emit_dot(lhs, rhs, builder, loc);
    }
}
