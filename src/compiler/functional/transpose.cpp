#include "syn/functional/transpose.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mln/sdk/functional/transpose.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "syn/operations.hpp"

namespace Syn::Compiler::Functional {
    mlir::Value Transpose::emit_typed(const Mln::Sdk::Functional::Transpose &transpose,
                                      Operations &ctx, mlir::OpBuilder &builder,
                                      mlir::Location loc) const {
        mlir::Value x = ctx.resolve(transpose.x, builder, loc);
        return mlir::stablehlo::TransposeOp::create(
                   builder, loc, x,
                   mlir::DenseI64ArrayAttr::get(builder.getContext(), transpose.permutation))
            .getResult();
    }
}
