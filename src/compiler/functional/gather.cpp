#include "syn/functional/gather.hpp"
#include "mln/sdk/functional/const.hpp"
#include "mln/sdk/functional/gather.hpp"
#include "mln/sdk/functional/map.hpp"
#include "syn/functional/map.hpp"
#include "syn/operations.hpp"

#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <vector>

namespace Syn::Compiler::Functional {
    namespace {
        mlir::Value emit_gather(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value table,
                                mlir::Value indices, int64_t axis) {
            auto table_type = mlir::cast<mlir::RankedTensorType>(table.getType());
            auto indices_type = mlir::cast<mlir::RankedTensorType>(indices.getType());
            auto table_shape = table_type.getShape();
            auto indices_shape = indices_type.getShape();
            const int64_t table_rank = table_type.getRank();
            const int64_t indices_rank = indices_type.getRank();
            const int64_t result_rank = table_rank - 1 + indices_rank;

            std::vector<int64_t> slice_sizes(table_shape.begin(), table_shape.end());
            slice_sizes[axis] = 1;
            std::vector<int64_t> offset_dimensions;
            for (int64_t i = 0; i < axis; ++i)
                offset_dimensions.push_back(i);
            for (int64_t i = axis + indices_rank; i < result_rank; ++i)
                offset_dimensions.push_back(i);

            auto dimensions = mlir::stablehlo::GatherDimensionNumbersAttr::get(
                builder.getContext(), offset_dimensions, std::vector<int64_t>{axis},
                std::vector<int64_t>{}, std::vector<int64_t>{}, std::vector<int64_t>{axis},
                indices_rank);

            std::vector<int64_t> result_shape;
            for (int64_t i = 0; i < axis; ++i)
                result_shape.push_back(table_shape[i]);
            result_shape.insert(result_shape.end(), indices_shape.begin(), indices_shape.end());
            for (int64_t i = axis + 1; i < table_rank; ++i)
                result_shape.push_back(table_shape[i]);

            return mlir::stablehlo::GatherOp::create(
                       builder, loc,
                       mlir::RankedTensorType::get(result_shape, table_type.getElementType()),
                       table, indices, dimensions, slice_sizes, false)
                .getResult();
        }
    }

    mlir::Value Gather::emit_typed(const Mln::Sdk::Functional::Gather &gather, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const {
        if (gather.table.def->kind == Mln::Sdk::Functional::Kind::Const) {
            const auto &constant =
                static_cast<const Mln::Sdk::Functional::Const &>(*gather.table.def);
            if (constant.quantize && gather.axis == 0) {
                Mln::Sdk::Tensor expanded = ctx.decode_weight(constant);
                if (const auto *map = row_addressable_map(expanded)) {
                    mlir::Value indices = ctx.resolve(gather.indices, builder, loc);
                    mlir::Value selected =
                        emit_gathered_map(*map, indices, gather.axis, ctx, builder, loc);
                    auto selected_type = mlir::cast<mlir::RankedTensorType>(selected.getType());
                    return mlir::stablehlo::ReshapeOp::create(
                               builder, loc,
                               mlir::RankedTensorType::get(gather.signature(),
                                                           selected_type.getElementType()),
                               selected)
                        .getResult();
                }
            }
        }
        return emit_gather(builder, loc, ctx.resolve(gather.table, builder, loc),
                           ctx.resolve(gather.indices, builder, loc), gather.axis);
    }
}
