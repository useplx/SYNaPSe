#include "syn/functional/select.hpp"

#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    mlir::Value Select::emit_typed(const Mln::Sdk::Functional::Select &select, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const {
        mlir::Value condition = ctx.resolve(select.condition, builder, loc);
        mlir::Value when_true = ctx.resolve(select.when_true, builder, loc);
        mlir::Value when_false = ctx.resolve(select.when_false, builder, loc);
        return mlir::stablehlo::SelectOp::create(builder, loc, condition, when_true, when_false)
            .getResult();
    }
}
