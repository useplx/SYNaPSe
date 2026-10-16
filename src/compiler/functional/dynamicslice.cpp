#include "syn/functional/dynamicslice.hpp"
#include "mln/sdk/functional/dynamicslice.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "syn/operations.hpp"
#include <vector>

namespace Syn::Compiler::Functional {
    mlir::Value DynamicSlice::emit_typed(const Mln::Sdk::Functional::DynamicSlice &dynamicslice,
                                         Operations &ctx, mlir::OpBuilder &builder,
                                         mlir::Location loc) const {
        mlir::Value operand = ctx.resolve(dynamicslice.x, builder, loc);

        std::vector<mlir::Value> start_indices;
        start_indices.reserve(dynamicslice.start_indices.size());
        for (const auto &idx : dynamicslice.start_indices)
            start_indices.push_back(ctx.resolve(idx, builder, loc));

        return mlir::stablehlo::DynamicSliceOp::create(
                   builder, loc, operand, mlir::ValueRange(start_indices), dynamicslice.slice_sizes)
            .getResult();
    }
}
