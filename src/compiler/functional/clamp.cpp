#include "syn/functional/clamp.hpp"

#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    mlir::Value Clamp::emit_typed(const Mln::Sdk::Functional::Clamp &node, Operations &ctx,
                                  mlir::OpBuilder &builder, mlir::Location loc) const {
        mlir::Value minimum = ctx.resolve(node.minimum, builder, loc);
        mlir::Value input = ctx.resolve(node.x, builder, loc);
        mlir::Value maximum = ctx.resolve(node.maximum, builder, loc);
        return mlir::stablehlo::ClampOp::create(builder, loc, minimum, input, maximum).getResult();
    }
}
