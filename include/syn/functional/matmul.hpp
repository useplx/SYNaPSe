#pragma once

#include "mln/sdk/functional/matmul.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class MatMul : public Operation::For<Mln::Sdk::Functional::MatMul> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::MatMul &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
