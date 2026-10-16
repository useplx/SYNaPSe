#include "syn/functional/equal.hpp"

#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    mlir::Value Equal::emit_typed(const Mln::Sdk::Functional::Equal &equal, Operations &ctx,
                                  mlir::OpBuilder &builder, mlir::Location loc) const {
        mlir::Value lhs = ctx.resolve(equal.lhs, builder, loc);
        mlir::Value rhs = ctx.resolve(equal.rhs, builder, loc);
        return mlir::stablehlo::CompareOp::create(builder, loc, lhs, rhs,
                                                  mlir::stablehlo::ComparisonDirection::EQ)
            .getResult();
    }
}
