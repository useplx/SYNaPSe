#include "syn/functional/bitcastconvert.hpp"
#include "mln/sdk/functional/bitcastconvert.hpp"
#include "syn/operations.hpp"

#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    mlir::Value BitcastConvert::emit_typed(const Mln::Sdk::Functional::BitcastConvert &bitcast,
                                           Operations &ctx, mlir::OpBuilder &builder,
                                           mlir::Location loc) const {
        mlir::Value operand = ctx.resolve(bitcast.x, builder, loc);
        auto result_type = mlir::RankedTensorType::get(
            bitcast.shape, dtype_to_mlir_type(bitcast.target, *builder.getContext()));
        return mlir::stablehlo::BitcastConvertOp::create(builder, loc, result_type, operand)
            .getResult();
    }
}
