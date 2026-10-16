#include "syn/functional/convert.hpp"
#include "mln/sdk/functional/convert.hpp"
#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    mlir::Value Convert::emit_typed(const Mln::Sdk::Functional::Convert &convert, Operations &ctx,
                                    mlir::OpBuilder &builder, mlir::Location loc) const {
        mlir::Value operand = ctx.resolve(convert.x, builder, loc);
        auto target_type = dtype_to_mlir_type(convert.target, *builder.getContext());
        return mlir::stablehlo::ConvertOp::create(builder, loc, operand, target_type).getResult();
    }
}
