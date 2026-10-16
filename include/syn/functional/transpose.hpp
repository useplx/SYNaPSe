#pragma once

#include "mln/sdk/functional/transpose.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {

    class Transpose : public Operation::For<Mln::Sdk::Functional::Transpose> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Transpose &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
