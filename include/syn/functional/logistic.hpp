#pragma once
#include "syn/operations.hpp"
namespace Syn::Compiler {
    mlir::Value emit_logistic(mlir::Operation &op, mlir::OpBuilder &builder, mlir::Location loc);
}
