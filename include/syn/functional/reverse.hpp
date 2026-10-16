#pragma once

#include "mln/sdk/functional/reverse.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Reverse : public Operation::For<Mln::Sdk::Functional::Reverse> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Reverse &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
