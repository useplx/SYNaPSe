#include "syn/functional/slice.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mln/error.hpp"
#include "mln/sdk/functional/const.hpp"
#include "mln/sdk/functional/map.hpp"
#include "mln/sdk/functional/slice.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "syn/dynamic_shape.hpp"
#include "syn/errors/graph.hpp"
#include "syn/functional/map.hpp"
#include "syn/operations.hpp"

#include <algorithm>
#include <stdexcept>

namespace Syn::Compiler::Functional {
    mlir::Value Slice::emit_typed(const Mln::Sdk::Functional::Slice &slice, Operations &ctx,
                                  mlir::OpBuilder &builder, mlir::Location loc) const {
        if (slice.x.def && slice.x.def->kind == Mln::Sdk::Functional::Kind::Const &&
            !slice.starts.empty() && slice.starts.size() == slice.limits.size() &&
            slice.starts.size() == slice.strides.size() &&
            slice.starts.size() == slice.x.shape.size() && slice.strides.front() == 1) {
            const auto &constant = static_cast<const Mln::Sdk::Functional::Const &>(*slice.x.def);
            bool row_slice = constant.quantize;
            for (std::size_t axis = 1; axis < slice.starts.size(); ++axis)
                row_slice = row_slice && slice.starts[axis] == 0 && slice.strides[axis] == 1 &&
                            slice.limits[axis] == slice.x.shape[axis];

            if (row_slice) {
                Mln::Sdk::Tensor expanded = ctx.decode_weight(constant);
                if (const auto *map = row_addressable_map(expanded)) {
                    const int64_t count = slice.limits.front() - slice.starts.front();
                    std::vector<int32_t> values(static_cast<std::size_t>(count));
                    for (int64_t index = 0; index < count; ++index)
                        values[static_cast<std::size_t>(index)] =
                            static_cast<int32_t>(slice.starts.front() + index);
                    const auto type = mlir::RankedTensorType::get({count}, builder.getI32Type());
                    mlir::Value indices =
                        mlir::stablehlo::ConstantOp::create(
                            builder, loc, mlir::DenseIntElementsAttr::get(type, values))
                            .getResult();
                    mlir::Value selected = emit_gathered_map(*map, indices, 0, ctx, builder, loc);
                    const auto selected_type =
                        mlir::cast<mlir::RankedTensorType>(selected.getType());
                    return mlir::stablehlo::ReshapeOp::create(
                               builder, loc,
                               mlir::RankedTensorType::get(slice.signature(),
                                                           selected_type.getElementType()),
                               selected)
                        .getResult();
                }
            }
        }

        mlir::Value x = ctx.resolve(slice.x, builder, loc);

        auto mlir_context = builder.getContext();
        if (std::find(slice.limits.begin(), slice.limits.end(), Mln::Sdk::Tensor::Dynamic) !=
            slice.limits.end()) {
            if (!slice.limits_source.def)
                raise(Errors::Graph::Slice::Limit::Source::Missing);
            mlir::Value source = ctx.resolve(slice.limits_source, builder, loc);
            std::vector<mlir::Value> limit_parts;
            std::size_t dynamic_index = 0;
            for (int64_t limit : slice.limits) {
                mlir::Value value;
                if (limit == Mln::Sdk::Tensor::Dynamic) {
                    value = dimension_size(builder, loc, source,
                                           slice.limits_source_dimensions.at(dynamic_index++));
                    value = mlir::stablehlo::ConvertOp::create(
                                builder, loc, mlir::RankedTensorType::get({}, builder.getI64Type()),
                                value)
                                .getResult();
                } else {
                    auto scalar = mlir::RankedTensorType::get({}, builder.getI64Type());
                    value =
                        mlir::stablehlo::ConstantOp::create(
                            builder, loc,
                            mlir::DenseElementsAttr::get(scalar, builder.getI64IntegerAttr(limit)))
                            .getResult();
                }
                limit_parts.push_back(
                    mlir::stablehlo::ReshapeOp::create(
                        builder, loc, mlir::RankedTensorType::get({1}, builder.getI64Type()), value)
                        .getResult());
            }
            auto vector_type = mlir::RankedTensorType::get(
                {static_cast<int64_t>(slice.limits.size())}, builder.getI64Type());
            mlir::Value limits =
                mlir::stablehlo::ConcatenateOp::create(builder, loc, vector_type, limit_parts, 0)
                    .getResult();
            mlir::Value starts =
                mlir::stablehlo::ConstantOp::create(
                    builder, loc, mlir::DenseIntElementsAttr::get(vector_type, slice.starts))
                    .getResult();
            mlir::Value strides =
                mlir::stablehlo::ConstantOp::create(
                    builder, loc, mlir::DenseIntElementsAttr::get(vector_type, slice.strides))
                    .getResult();
            auto input_type = mlir::cast<mlir::RankedTensorType>(x.getType());
            auto result_type = mlir::RankedTensorType::get(mlir_shape(slice.signature()),
                                                           input_type.getElementType());
            return mlir::stablehlo::RealDynamicSliceOp::create(builder, loc, result_type, x, starts,
                                                               limits, strides)
                .getResult();
        }
        return mlir::stablehlo::SliceOp::create(
                   builder, loc, x, mlir::DenseI64ArrayAttr::get(mlir_context, slice.starts),
                   mlir::DenseI64ArrayAttr::get(mlir_context, slice.limits),
                   mlir::DenseI64ArrayAttr::get(mlir_context, slice.strides))
            .getResult();
    }
}
