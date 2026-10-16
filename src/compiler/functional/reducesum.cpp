#include "syn/functional/reducesum.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mln/sdk/functional/reducesum.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "syn/dynamic_shape.hpp"
#include "syn/operations.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <vector>

namespace Syn::Compiler::Functional {
    namespace {
        mlir::Value emit_reduce(mlir::OpBuilder &builder, mlir::Location location,
                                mlir::Value input, mlir::Type element,
                                const std::vector<int64_t> &axes,
                                const std::vector<int64_t> &result_shape) {
            const auto result_type = mlir::RankedTensorType::get(result_shape, element);
            const auto scalar_type = mlir::RankedTensorType::get({}, element);
            mlir::Value initial =
                mlir::stablehlo::ConstantOp::create(
                    builder, location,
                    mlir::DenseElementsAttr::get(scalar_type, builder.getZeroAttr(element)))
                    .getResult();
            auto reduction = mlir::stablehlo::ReduceOp::create(
                builder, location, mlir::TypeRange{result_type}, mlir::ValueRange{input},
                mlir::ValueRange{initial},
                mlir::DenseI64ArrayAttr::get(builder.getContext(), axes));

            mlir::OpBuilder::InsertionGuard guard(builder);
            mlir::Block *block = builder.createBlock(
                &reduction.getBody(), {}, {scalar_type, scalar_type}, {location, location});
            builder.setInsertionPointToStart(block);
            auto sum = mlir::stablehlo::AddOp::create(builder, location, block->getArgument(0),
                                                      block->getArgument(1));
            mlir::stablehlo::ReturnOp::create(builder, location, sum.getResult());
            return reduction.getResult(0);
        }

        mlir::Value scalar_i32(mlir::OpBuilder &builder, mlir::Location location, int32_t value) {
            const auto type = mlir::RankedTensorType::get({}, builder.getI32Type());
            return mlir::stablehlo::ConstantOp::create(
                       builder, location,
                       mlir::DenseElementsAttr::get(type, builder.getI32IntegerAttr(value)))
                .getResult();
        }

        mlir::Value emit_dynamic_tiled_reduce(mlir::OpBuilder &builder, mlir::Location location,
                                              mlir::Value input, mlir::Type element, int64_t axis,
                                              int64_t tile, const std::vector<int64_t> &shape,
                                              const std::vector<int64_t> &axes,
                                              const std::vector<int64_t> &result_shape) {
            std::vector<int64_t> low(shape.size(), 0);
            std::vector<int64_t> high(shape.size(), 0);
            std::vector<int64_t> interior(shape.size(), 0);
            high[axis] = tile;

            const auto scalar_type = mlir::RankedTensorType::get({}, element);
            mlir::Value zero =
                mlir::stablehlo::ConstantOp::create(
                    builder, location,
                    mlir::DenseElementsAttr::get(scalar_type, builder.getZeroAttr(element)))
                    .getResult();
            mlir::Value padded =
                mlir::stablehlo::PadOp::create(builder, location, input, zero, low, high, interior)
                    .getResult();

            mlir::Value extent = dimension_size(builder, location, input, axis);
            mlir::Value chunks =
                mlir::stablehlo::DivOp::create(
                    builder, location,
                    mlir::stablehlo::AddOp::create(
                        builder, location, extent,
                        scalar_i32(builder, location, static_cast<int32_t>(tile - 1))),
                    scalar_i32(builder, location, static_cast<int32_t>(tile)))
                    .getResult();

            std::vector<int64_t> slice_sizes(shape.begin(), shape.end());
            slice_sizes[axis] = tile;
            const auto slice = [&](mlir::Value index) {
                std::vector<mlir::Value> starts(shape.size(), scalar_i32(builder, location, 0));
                starts[axis] = mlir::stablehlo::MulOp::create(
                                   builder, location, index,
                                   scalar_i32(builder, location, static_cast<int32_t>(tile)))
                                   .getResult();
                return mlir::stablehlo::DynamicSliceOp::create(
                           builder, location, padded, mlir::ValueRange(starts), slice_sizes)
                    .getResult();
            };

            mlir::Value index = scalar_i32(builder, location, 0);
            mlir::Value result =
                emit_reduce(builder, location, slice(index), element, axes, result_shape);
            index = scalar_i32(builder, location, 1);

            const std::array<mlir::Type, 2> result_types{index.getType(), result.getType()};
            const std::array<mlir::Value, 2> operands{index, result};
            auto loop = mlir::stablehlo::WhileOp::create(builder, location, result_types, operands);

            mlir::OpBuilder::InsertionGuard guard(builder);
            mlir::Block &condition = loop.getCond().emplaceBlock();
            condition.addArguments(result_types, std::array<mlir::Location, 2>{location, location});
            builder.setInsertionPointToStart(&condition);
            mlir::Value predicate =
                mlir::stablehlo::CompareOp::create(builder, location, condition.getArgument(0),
                                                   chunks, mlir::stablehlo::ComparisonDirection::LT,
                                                   mlir::stablehlo::ComparisonType::SIGNED)
                    .getResult();
            mlir::stablehlo::ReturnOp::create(builder, location, predicate);

            mlir::Block &body = loop.getBody().emplaceBlock();
            body.addArguments(result_types, std::array<mlir::Location, 2>{location, location});
            builder.setInsertionPointToStart(&body);
            mlir::Value part = emit_reduce(builder, location, slice(body.getArgument(0)), element,
                                           axes, result_shape);
            mlir::Value next =
                mlir::stablehlo::AddOp::create(builder, location, body.getArgument(1), part)
                    .getResult();
            mlir::Value next_index =
                mlir::stablehlo::AddOp::create(builder, location, body.getArgument(0),
                                               scalar_i32(builder, location, 1))
                    .getResult();
            mlir::stablehlo::ReturnOp::create(builder, location,
                                              mlir::ValueRange{next_index, next});
            return loop.getResult(1);
        }
    }

    mlir::Value ReduceSum::emit_typed(const Mln::Sdk::Functional::ReduceSum &reducesum,
                                      Operations &ctx, mlir::OpBuilder &builder,
                                      mlir::Location loc) const {
        mlir::Value x = ctx.resolve(reducesum.x, builder, loc);
        const auto &axes = reducesum.axes;

        auto x_type = mlir::cast<mlir::RankedTensorType>(x.getType());
        auto storage_element = x_type.getElementType();
        auto elem = mlir::isa<mlir::Float16Type>(storage_element)
                        ? static_cast<mlir::Type>(mlir::Float32Type::get(builder.getContext()))
                        : storage_element;
        if (elem != storage_element) {
            auto accumulation_type = mlir::RankedTensorType::get(x_type.getShape(), elem);
            x = mlir::stablehlo::ConvertOp::create(builder, loc, accumulation_type, x).getResult();
        }
        auto shape = x_type.getShape();

        std::vector<int64_t> result_shape;
        for (std::size_t i = 0; i < shape.size(); i++) {
            bool reduced = false;
            for (int64_t axis : axes)
                reduced = reduced || axis == static_cast<int64_t>(i);
            if (!reduced)
                result_shape.push_back(shape[i]);
        }
        mlir::Value result;
        const bool static_shape =
            std::ranges::all_of(shape, [](int64_t dimension) { return dimension > 0; });
        if (static_shape) {
            int64_t budget = std::numeric_limits<int64_t>::max();
            bool tiled = false;
            for (const int64_t axis : axes) {
                const int64_t extent = shape[axis];
                const int64_t requested = ctx.tile(reducesum, axis, extent);
                if (requested > 0 && requested < extent) {
                    budget = std::min(budget, requested);
                    tiled = true;
                }
            }

            if (tiled) {
                std::vector<int64_t> tiles(shape.begin(), shape.end());
                int64_t remaining = budget;
                for (const int64_t axis : axes) {
                    tiles[axis] = std::min(shape[axis], std::max<int64_t>(remaining, 1));
                    remaining = std::max<int64_t>(remaining / tiles[axis], 1);
                }

                std::vector<int64_t> starts(shape.size(), 0);
                std::vector<int64_t> strides(shape.size(), 1);
                std::function<void(std::size_t)> emit_parts = [&](std::size_t index) {
                    if (index != axes.size()) {
                        const int64_t axis = axes[index];
                        for (int64_t begin = 0; begin < shape[axis]; begin += tiles[axis]) {
                            starts[axis] = begin;
                            emit_parts(index + 1);
                        }
                        starts[axis] = 0;
                        return;
                    }

                    std::vector<int64_t> limits(shape.begin(), shape.end());
                    for (const int64_t axis : axes)
                        limits[axis] = std::min(starts[axis] + tiles[axis], shape[axis]);
                    mlir::Value part = mlir::stablehlo::SliceOp::create(
                        builder, loc, x, mlir::DenseI64ArrayAttr::get(builder.getContext(), starts),
                        mlir::DenseI64ArrayAttr::get(builder.getContext(), limits),
                        mlir::DenseI64ArrayAttr::get(builder.getContext(), strides));
                    part = emit_reduce(builder, loc, part, elem, axes, result_shape);
                    result =
                        result
                            ? mlir::stablehlo::AddOp::create(builder, loc, result, part).getResult()
                            : part;
                };
                emit_parts(0);
            }
        }
        if (!result && axes.size() == 1) {
            const int64_t axis = axes.front();
            const bool supported_dynamic_shape =
                shape[axis] < 0 && std::ranges::all_of(
                                       shape,
                                       [axis, index = int64_t{0}](int64_t dimension) mutable {
                                           return index++ == axis || dimension > 0;
                                       });
            const int64_t tile = ctx.tile(reducesum, axis, shape[axis]);
            if (supported_dynamic_shape && tile > 0)
                result = emit_dynamic_tiled_reduce(builder, loc, x, elem, axis, tile,
                                                   std::vector<int64_t>(shape.begin(), shape.end()),
                                                   axes, result_shape);
        }
        if (!result)
            result = emit_reduce(builder, loc, x, elem, axes, result_shape);

        if (elem != storage_element) {
            auto storage_result_type = mlir::RankedTensorType::get(result_shape, storage_element);
            return mlir::stablehlo::ConvertOp::create(builder, loc, storage_result_type, result)
                .getResult();
        }
        return result;
    }
}
