#include "syn/functional/dynamicupdateslice.hpp"
#include "mln/sdk/functional/dynamicupdateslice.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "syn/operations.hpp"
#include <vector>

namespace Syn::Compiler::Functional {
    mlir::Value
    DynamicUpdateSlice::emit_typed(const Mln::Sdk::Functional::DynamicUpdateSlice &update,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
        mlir::Value operand = ctx.resolve(update.x, builder, loc);
        mlir::Value update_value = ctx.resolve(update.update, builder, loc);

        std::vector<mlir::Value> start_indices;
        start_indices.reserve(update.start_indices.size());
        for (const auto &idx : update.start_indices)
            start_indices.push_back(ctx.resolve(idx, builder, loc));

        return mlir::stablehlo::DynamicUpdateSliceOp::create(builder, loc, operand, update_value,
                                                             mlir::ValueRange(start_indices))
            .getResult();
    }
}
