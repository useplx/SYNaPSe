#include "syn/functional/reverse.hpp"

#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    mlir::Value Reverse::emit_typed(const Mln::Sdk::Functional::Reverse &node, Operations &ctx,
                                    mlir::OpBuilder &builder, mlir::Location loc) const {
        mlir::Value input = ctx.resolve(node.x, builder, loc);
        return mlir::stablehlo::ReverseOp::create(builder, loc, input, node.dimensions).getResult();
    }
}
